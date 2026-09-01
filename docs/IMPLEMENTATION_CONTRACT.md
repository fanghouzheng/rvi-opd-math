# 实现契约

## 1. 母工程与边界

GPU 主实现固定在 Relay-OPD commit `eab2145...` 内完成，因为它已提供 Qwen3-4B→1.7B、DAPO、`verl`、vLLM speculative decoding、八套数学评测和全部主要基线。RvI 代码以小补丁接入，不重新维护一份 `verl`。

TA-OPD 的 slime 实现只作为信号计算参考；其 `D`、`C` 公式需移植到 Relay 的 rollout scorer。TRD 优先调用 Relay 已带的 TRD baseline；原 TRD repo 用于核对 rewrite、FKL 和日志，不混入其 DeepScaleR/LoRA 配方。

## 2. 每个 rollout state 的强制字段

记录格式必须满足 `schemas/state_record.schema.json`，并额外持久化：

- teacher/student top-K token id 与 log-prob；
- `D_raw, C_raw, D_norm, C_norm, D_L, D_I, s2_raw, s2_norm`；
- 阈值版本、候选状态 rank、动作、动作接受/回滚原因；
- trigger 前缀哈希、problem id、rollout seed、token position；
- teacher leg 长度、repair window 长度和三本预算账；
- bridge 后 H-token probe 的 s2 AUC 与 teacher-preferred rate。

禁止默认保存可逆的完整训练文本到公开 artifact；文本样例需单独脱敏和抽样。

## 3. 信号计算

1. `D`：师生 top-K 并集上重整化后的 teacher→student FKL。
2. `C`：教师在学生 top-K support 上的原始总质量；若只拿到 teacher top-K，则另名 `C_lower_bound`，不能与精确 C 混报。
3. `D_L = norm(D) * norm(C)`；`D_I = norm(D) * (1 - norm(C))`。
4. `s2`：teacher softmax 在 epistemic-onset token id 集合上的质量，温度固定 1.0，在 KL temperature scaling 前计算。
5. 16 个短语分别以 bare/leading-space tokenization 取首 sub-token 后去重；每个 tokenizer/version 都保存最终 id 列表。
6. 5–95% robust normalization 只在当前训练 batch 内用于在线路由；论文阈值由 D1 calibration state bank 单独冻结。

## 4. 动作语义

`repair` 不改上下文。原 student trajectory 保留，在指定原状态位置施加 teacher top-128 重整化 forward CE。D0 的强局部修复使用与 paired bridge 实际 token 数相同的连续原轨迹窗口；窗口不足的 state 标为 censored，不进入“逐 state 严格匹配”主分析。

`intervene` 从触发位置开始插入 teacher top-1 epistemic token，再生成 L=3 个以双换行分隔的段落；最多 M=2 次。主方法在未耗尽 M 时交回 student；D0 paired probe 不因 M 用尽提前截断，以免把 action 与 length 混淆。

`discard` 保留 token 作为上下文但不产生直接蒸馏 loss。

## 5. 验收与回滚

验收门使用 rollout snapshot，不能使用更新后的 actor。bridge 后由 snapshot student 续写 H=128 token：若 `s2_post <= D1.s2_low`，或 teacher-preferred token rate 相对 bridge 前 32-token window 提升，则接受。否则恢复触发前 KV/cache 状态，丢弃 bridge，按 repair 路径重新生成原上下文。必须记录被丢弃 bridge 的生成成本。

严格 D2 另做 paired base continuation，使用 common random numbers；在线训练不为每次动作额外生成 paired base，以控制成本。

## 6. detached 对照

detached arm 对 teacher bridge token 计算与 intervene 相同的监督 loss，但后续 student continuation 从原 student prefix 开始，bridge token 不写入 KV/cache。其 teacher-owned token 数、监督位置数、teacher logits 调用与 intervene 一致。

## 7. 预算与声明

- `teacher_generated_tokens`：teacher 实际自回归生成并拥有的 token；失败 bridge 也计入成本。
- `supervised_positions`：直接参与蒸馏损失的位置数。
- `teacher_forward_tokens`：teacher 被评分/生成的输入输出 token 等价量。

D0 主机制对照按 `supervised_positions` 逐 state 精确匹配，并同时报告另外两账。只有对应 ledger 相等时才称该 ledger “budget matched”；不得把监督预算写成 wall-clock 或 teacher generation 预算。

## 8. 防泄漏和可复现

- calibration split 在 dedup 后从训练 prompt 产生，不含任何评测题或近重复题。
- 每次运行写入 git commit、upstream commit、完整 Hydra/YAML、镜像 digest、GPU 型号、CUDA/vLLM/verl 版本和数据哈希。
- checkpoint selection 仅依据预注册 validation composite，不按 test suite 平均分挑 step。
- 训练 seed 至少 3 个；如资源不足，单 seed 只标记 pilot，不进入主结论。

