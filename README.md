# RvI-OPD Math

RvI-OPD（Repair-vs-Intervene On-Policy Distillation）的数学域实验仓库。项目研究同一类教师-学生分歧在不同前缀状态下应接受局部分布修复（`repair`, d=0）还是轨迹桥接（`intervene`, d=L），并把这一主张拆成可证伪、预算可审计的实验。

当前仓库是第一阶段研究包：包含冻结协议、实验矩阵、统一配置、核心信号/路由/预算代码、统计契约和上游代码固定脚本。GPU 训练以 Relay-OPD 的 `verl + vLLM 0.21.0` 实现为母工程；本仓库不复制整份 `verl`。

## 结论先行

- 主模型对：`Qwen3-4B-Instruct-2507 -> Qwen3-1.7B-Non-Thinking`。
- 训练集：`DAPO-Math-17K` 英文子集，一轮训练，严格冻结数据顺序与 prompt。
- 主评测：AIME24/25/26、MATH500、AMC23、OlympiadBench、HMMT-Feb26、HMMT-Nov25；MMLU 只做遗忘检查。
- 先做 D1 信号定标和 D2 冻结状态机制门，再进入完整训练。机制门不通过时不宣称 state-dependent routing。
- 主要比较同时报告 teacher-generated tokens、direct-supervision positions 和 teacher-forward-equivalent tokens；三者不能混称“同预算”。

完整协议见 [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)，工程接口见 [docs/IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md)，服务器部署见 [docs/REMOTE_RUNBOOK.md](docs/REMOTE_RUNBOOK.md)。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python scripts/doctor.py configs/base.yaml
python -m rvi_opd.cli demo --seed 7
```

下载固定版本的三个参考实现：

```bash
bash scripts/bootstrap_upstreams.sh
```

GPU 训练前需在 Linux/NVIDIA 环境中按 Relay-OPD 上游说明安装 CUDA、PyTorch、`verl` 和 `vLLM==0.21.0`。本地 macOS 只用于协议编辑、单测、数据审计和结果聚合。

## 仓库结构

```text
configs/                    冻结配置和各阶段实验配置
docs/                       完整实验计划与实现契约
experiments/manifest.csv    运行注册表；每次 GPU run 必须占一行
schemas/                    状态库与结果 JSON Schema
scripts/                    环境检查和上游固定脚本
src/rvi_opd/                信号、路由、预算与统计核心
tests/                      CPU 单元测试
```

## 复现原则

1. 先注册再运行：不得在看过测试集结果后补写 run id 或改阈值。
2. D1 只用训练集切出的 calibration split；测试题绝不参与阈值选择。
3. 所有动作对照共享 prompt、checkpoint、候选状态和随机种子。
4. 置信区间按 problem cluster bootstrap；同题多次采样不是独立样本。
5. 失败门和负结果原样保留，不用后续超参搜索覆盖。

## 上游版本

| 项目 | commit |
|---|---|
| Relay-OPD | `eab21451f99e1a40fbb244f556de766d153c88f5` |
| TRD | `5f3894d776cb2b762a44e09f8ce8293a762e21af` |
| TA-OPD | `ccdf21d2066466f3d616f63cd867cc49119c45e6` |

仓库暂不附开源许可证；如需公开发布，应先确认三方上游代码与模型/数据许可，再单独选择许可证。
