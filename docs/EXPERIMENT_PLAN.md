# RvI-OPD 数学域完整实验计划（v0.1）

## 1. 研究问题与可证伪假设

核心问题不是“哪些 token 值得监督”，而是“给定一个被标记状态，应保持原上下文做局部 repair，还是插入 teacher leg 改变后续状态”。

预注册假设：

- H1（异质性）：在 `D_I` 高但 s2 低的状态上，强 repair 不劣于 intervene；off-support 本身不等于不可吸收。
- H2（状态损坏）：在 `D_I` 高且 s2 高的状态上，intervene 的下游 s2 残留和 verifier 结果优于 budget-matched repair，尽管两者都能改善被修位置的局部分布。
- H3（context 因果）：bridge 正常进入后续 context 优于 detached bridge；若二者相同，收益主要来自额外 target，而非状态改变。
- H4（路由价值）：真实路由优于动作比例相同、但跨状态随机打乱的路由；若不优于，方法只能作为组件组合而非 state-dependent routing 贡献。

停止规则：D2 若不能同时满足“repair 的 s2 残留无显著下降”和“bridge 的 s2 残留显著下降”，停止 H2/H3 的确认性措辞；仍可完成探索性主表，但论文定位降级。

## 2. 统一母协议

采用 Relay-OPD 的公开主设置，避免混用三篇论文各自训练 recipe。

| 项 | 冻结值 |
|---|---|
| Teacher | Qwen3-4B-Instruct-2507 |
| Student | Qwen3-1.7B Non-Thinking |
| Train | DAPO-Math-17K English，去除评测近重复 |
| Prompt/response | 2,048 / 16,384 tokens |
| Rollout | temperature=1.0, top-p=1.0, n=1 |
| Optimizer | paper-aligned OPD objective, LR=1e-6 constant, batch=128, 1 epoch |
| Relay | K=5, M=2, L=3 paragraphs |
| Repair | teacher top-128 renormalized FCE, original context |
| Signal support | K=16 主值；8/32 只做 audit |
| Train seeds | 17, 29, 43 |
| Hardware reference | 8×H100；pilot 可 2/4 GPU 但不进主表 |

TRD 在本协议内使用相同 DAPO prompt、模型、长度、训练步和评测；不直接引用原论文 DeepScaleR+LoRA 数字作为同表可比结果。

## 3. 数据与切分

1. 标准化 DAPO 英文子集，保留原 problem id、source、answer、reward metadata。
2. 对八个评测集做 exact、规范化文本、MinHash/embedding 三层去重；疑似近重复人工复核。
3. 从去重后的训练集按 source 分层留出 5% calibration split，仅用于 D1 和阈值冻结；剩余用于训练。
4. 另取 512 个 held-out train problems 形成 fixed-state mechanism bank，不用于任何权重更新。
5. 所有 split 输出 SHA256 和行数审计；评测答案不得进入 prompt 或 router。

## 4. 阶段与决策门

### P0：基础复现与最小烟测

目的：确认母工程、数据、verifier、prompt 与论文一致。

- 复现 untrained student 和 teacher 的八基准均值；允许与论文差异 ±1.0pp，超出先查模型 revision、prompt 和 sampling。
- 跑 128 prompt、max response 2,048 的 OPD/Relay smoke，确认无 NaN、预算日志守恒、桥接可交还 student。
- 用 20 个手工 state 校验 top-K、C、D、s2 与离线脚本逐 token 一致，容差 1e-5。

### D1：信号定标

在 calibration split 上采 512 条 student rollout，每条最多均匀保留 256 个状态，按 normalized position 八分箱。

输出：D/C/s2 分布；DL/DI 与 s2 的相关矩阵；候选状态覆盖率；K∈{8,16,32} 稳定性；TRD16 与 Relay13 反思集合敏感性；阈值 JSON。阈值主值为 repair score q90、s2 q90；s2-low 定义为 ≤q50。

通过条件：每个确认性 state stratum 至少 2,000 states、至少 200 unique problems；否则降低 high quantile 到 q85，一次性记录，不继续搜索。

### D2：三臂 paired continuation 机制硬门

从 held-out state bank 固定 512 个 `D_I-high & s2-high` 状态。三臂使用同一前缀：

1. base checkpoint 从原前缀续写；
2. repair-trained checkpoint 从原前缀续写；
3. base checkpoint 从“原前缀 + teacher bridge”续写。

每 state 8 个 completion，H=128，采用 common random numbers。主指标是 H-token teacher s2 residual AUC；次指标是 teacher-preferred token rate 和可在短窗判定时的 verifier pass。按 problem cluster 做 paired bootstrap。

硬门：`CI95(Δs2_repair)` 包含 0 或更差，同时 `CI95(Δs2_bridge)` 完全小于 0。verifier 是支持证据，不作为硬门，因为 128 token 未必完成解题。

### D0：state × action 析因实验

完整诊断为 2×2×2：support type (`D_L-high`, `D_I-high`) × state damage (`s2-low`, `s2-high`) × action (`repair`, `intervene`)。同一候选 state 先生成一次 paired teacher bridge，记录实际长度 b；repair 在原 student trajectory 从 trigger 起监督 b 个位置，intervene 插入这 b 个 teacher-owned token。原轨迹余量不足 b 的 state 标为 censored，主分析剔除、敏感性分析保留 IPW。

主展示四格：`D_I-high/s2-low` 的 repair vs intervene，以及 `D_I-high/s2-high` 的 repair vs intervene。DL 四格用于验证一般性和“intervene 浪费”边界。

主要终点：

- local：trigger/window 上 teacher→student KL 变化；
- behavioral：paired continuation 的 s2 AUC、teacher-preferred rate、verifier pass；
- training：完整八基准 accuracy macro-average；
- efficiency：三本预算账、平均训练长度、tokens/s、wall-clock、峰值显存。

确认性统计是 action × s2 interaction；单格 p 值只作分解。报告 effect、problem-cluster bootstrap CI 和 Benjamini-Hochberg 校正后的 q 值。

### D3：detached context 对照

三臂：正常 bridge、bridge supervision 但不写入后续 KV/cache、original-context repair。正常与 detached 两臂逐 state 复用同一 teacher token，匹配 teacher-generated tokens、supervised positions 和 teacher-forward calls。

主对比为 normal bridge − detached。若不显著，不能声称收益来自 context 改变。

### E1：完整数学主表

分两层，避免一开始烧完全部算力。

Tier 1 必跑：Base、Vanilla OPD、FastOPD@4096、TA-OPD@5%、Relay-OPD、RvI-OPD、RvI-random-route、oracle/offline upper bound。三 seed。

Tier 2 在 RvI 通过 D2/D0 后跑：SFT、SeqKD、TIP-select、SKD、TRD，以及 FastOPD 1K/2K/8K sweep。TRD 是 d=∞ 参照，不与在线 teacher-token ledger 强行等价。

评测采样严格沿 Relay：AIME/AMC/HMMT 每题 32，MATH500/Olympiad 每题 4，temperature=1.0、top-p=1.0、max response=32,768。报告 mean accuracy、pass@k（预注册 k）、响应长度、invalid/timeout 率。模型选择只看 held-out validation composite。

### A：消融

- A1：去 s1、去 s2、只 DL、只 DI；
- A2：动作比例完全相同但按 problem 内随机打乱；这是 H4 的直接检验；
- A3：无 acceptance gate、gate 仅 s2、gate 仅 teacher-preferred；
- A4：M∈{1,2,3}，L∈{1,3,6}，只在 seed17；
- A5：support K∈{8,16,32}；
- A6：repair loss = top-K FCE / full-vocab FKL / RKL reweight；
- A7：阈值 q85/q90/q95，仅做稳健性，不据此挑主值；
- A8：重复率旁路信号，用于 s1/s2 双低的退化循环盲区。

## 5. 路由与验收算法

对每个 student state 计算 `D_L, D_I, s2`。优先级必须固定：

```text
if s2 >= tau_intervene and max(D_L, D_I) >= tau_candidate:
    propose intervene
    if bridge probe accepted: keep bridge
    else: rollback and repair original context
elif max(D_L, D_I) >= tau_repair:
    repair
else:
    discard direct loss (context retained)
```

online threshold 只读取 D1 冻结文件。front-loading 采用首次有效 trigger 优先，M=2；不得用 verifier、answer correctness 或 test performance 参与路由。

## 6. 统计计划

- 单位：problem，不是 token 或 completion。
- 主表：每训练 seed 先聚合每题 completion，再跨 seed 报 mean±sd；方法差异用 problem-cluster、seed-stratified paired bootstrap 10,000 次。
- 机制 state bank：按 problem cluster bootstrap，state/continuation 嵌套在 problem 内。
- primary family：H1、H2、H3、H4 四项，BH-FDR 0.05。
- 预注册最小有意义效应：主表 macro-average +1.0pp；s2 AUC 相对下降 20%；预算效率 +10%。小于阈值即使显著也称“统计显著但实践意义不足”。
- 同时报告 raw effect、95% CI、q value、样本量和缺失/censoring 比例。

## 7. 算力与运行数量

最小确认性包：P0 3 runs；D1 1 bank；D2 3 checkpoint/probe arms；D0 4 个主训练 cell×3 seeds；D3 3 arms×3 seeds；E1 Tier1 中与前述不重复的约 4 methods×3 seeds。预计约 27–33 个 8×H100 训练/评测任务，外加 state-bank scoring。按 Relay 报告的 35–55 best steps，先用 seed17 做 go/no-go 可显著降本。

如只有 4×H100，保持 global batch 128，用梯度累积，不改有效 batch；wall-clock 另报。2-GPU 仅做 smoke，不进入论文表。

## 8. 结果表模板

每行至少包含：method、seed、checkpoint step、八基准、macro avg、MMLU、mean response length、teacher-generated tokens、supervised positions、teacher-forward tokens、gate acceptance、rollback rate、train GPU-hours、eval GPU-hours、peak memory、git/data/config hash。

机制图固定三面板：局部 KL、H-token s2 residual AUC、verifier/pass；每个点按同一 problem 连线，禁止只画无配对柱状图。

## 9. 风险与降级

- s2 稀有：按 q85 降一次；仍不足则只报告连续变量 interaction，不做硬分箱。
- teacher 也被错误前缀拖走：记录 teacher answer/verifier 与 s2；该区域不作 intervene 成功案例。
- repair 窗口被 EOS 截断：主分析 censor，报告比例；不得用更短 bridge 偷换预算。
- acceptance gate 成本过高：在线用绝对阈值 fast gate，严格 paired gate 留在 D2。
- D2 不过：停止“需要新状态”的强主张，保留预算化 action ensemble 作为探索性结果。
- A2 不过：停止“state-dependent router”创新主张，方法降级为 Relay+TA 的组合基线。

## 10. 执行顺序

`P0 -> D1 threshold freeze -> D2 hard gate -> D0 seed17 -> D0 seeds29/43 -> D3 -> E1 Tier1 -> A2 -> E1 Tier2 -> remaining ablations`。

任何后续变更必须在 `experiments/manifest.csv` 增加新 run id，并在结果中区分 confirmatory 与 exploratory。

