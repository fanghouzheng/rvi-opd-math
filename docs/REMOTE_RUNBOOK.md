# 远程运行手册

## CPU 控制面

```bash
git clone git@github.com:fanghouzheng/rvi-opd-math.git
cd rvi-opd-math
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
python scripts/doctor.py configs/base.yaml
bash scripts/bootstrap_upstreams.sh
```

`doctor.py` 的 `gpu_training_ready=false` 在登录节点或 macOS 上是正常结果。三份上游会按论文核对过的 commit 下载到被 git 忽略的 `vendor/`。

## GPU 节点最低条件

- Linux x86_64；NVIDIA H100 80GB 优先；
- Python 3.12；
- Relay-OPD 要求的 CUDA/PyTorch 组合；
- `vLLM==0.21.0`；
- 8 GPU 主实验，4 GPU 可保持 batch 做较慢复现；2 GPU 仅 smoke。

安装 Relay 母工程：

```bash
cd vendor/relay-opd/relay-opd
python -m pip install -c environment/vllm-constraints.txt vllm==0.21.0
python -m pip install -e .
python -m pip install -r requirements-relay-opd.txt
python environment/verify_install.py
```

## 数据与路径约定

在共享存储上准备：

```text
$RVI_DATA_ROOT/train/dapo_math_17k_en.parquet
$RVI_DATA_ROOT/calibration/dapo_calibration.parquet
$RVI_DATA_ROOT/eval/<benchmark>.parquet
$RVI_MODEL_ROOT/Qwen3-4B-Instruct-2507/
$RVI_MODEL_ROOT/Qwen3-1.7B/
$RVI_RUN_ROOT/<run_id>/
```

不要把模型、parquet、checkpoint 或原始 rollout 提交到 Git。每个 `$RVI_RUN_ROOT/<run_id>` 至少保存：

```text
resolved_config.yaml
environment.json
data_hashes.json
metrics.jsonl
state_records.jsonl.zst
checkpoints/
```

## 运行纪律

1. 从 `experiments/manifest.csv` 领取一个 `run_id`，把状态改为 `running` 后提交该小改动。
2. 只使用该行指定的 config/seed，不在命令行静默覆盖确认性超参。
3. 运行前记录 `git rev-parse HEAD`、三个 upstream commit、容器 digest 和数据 SHA256。
4. 完成后先写原始结果，再运行聚合；不得手改聚合 JSON。
5. 失败任务标为 `failed` 并保留错误摘要；重跑使用新 attempt id，不覆盖原目录。

## 当前实现状态

本版本已完成实验协议、CPU 信号/路由/预算/统计模块、配置与 CI。Relay GPU 母工程中的 RvI rollout patch 尚未合入；在该补丁完成前，只能运行上游 Base/OPD/FastOPD/Relay/TRD 复现和本仓库 CPU 验证，不能启动 RvI 主训练。实现必须遵循 `docs/IMPLEMENTATION_CONTRACT.md`，尤其是三本预算账和 rollback 语义。
