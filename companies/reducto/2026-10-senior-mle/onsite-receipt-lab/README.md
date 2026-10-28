# Receipt extraction：一天的 MLE onsite 模拟项目

先读 [BRIEF.md](BRIEF.md)。这里提供现有代码和资源；你的工作是做数据审查、baseline、训练实验、改进、评估和演示。没有预填改进结果。

采用 **Qwen2.5-VL-3B-Instruct + CORD-v2 + Hugging Face TRL/PEFT**。来源与许可见 [SOURCES.md](SOURCES.md)，原版训练脚本在 [upstream/sft_vlm.py](upstream/sft_vlm.py)。这是模拟练习，不是已确认的 Reducto 真题。

## 推荐入口：Colab

打开 [notebooks/colab.ipynb](notebooks/colab.ipynb)，上传随本项目提供的 `reducto-onsite-receipt-lab.zip`。Notebook 包含安装、GPU 检查、下载、baseline、训练和评估单元格；按顺序执行，看到结果后再开始优化。

也可以在 GPU 机器上直接运行下面的命令。所有命令的工作目录都是本文件所在目录。

## 资源与安装

- Python 3.11/3.12；CUDA GPU；建议至少 25GB 空闲磁盘用于权重、数据、缓存和 adapter。
- 真实基础权重两个 safetensors 分片合计约 7.51 GB。4-bit 是加载时量化，不会把原始下载变成 2GB。
- 默认单卡 T4 16GB 的保守配置是起点，并未在当前 CPU 云环境验证 GPU 显存或性能；先跑 2 steps 测量再扩大。
- 不要把多张 GPU 显存相加理解为单卡容量。默认只用一张；第二张可用于独立实验。未提供多卡 DDP 的已验证配置。

新建独立环境后安装（Colab 已有环境则参考 notebook）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
python -m pytest -q tests
```

依赖版本是本项目固定的组合，不是建议升级到任何 latest。使用其他 CUDA/Colab 镜像时，先确认驱动兼容性与 torch/torchvision 匹配。Notebook 修改已加载依赖后需要重启 Python session。

CPU 验证环境用 PyTorch 的 CPU wheel、`requirements.txt`，只支持下面的 tiny-model 流程验证，不用于真实 QLoRA 训练。

## 数据和模型准备（计时前完成）

```bash
python -m receipt_lab.prepare --output data/cord
python -m receipt_lab.prefetch
```

数据脚本下载固定 revision 的全部官方 splits，导出 PNG 和 JSONL。输出目录已有 manifest 时拒绝覆盖；需要重新准备时使用新目录。检查 `data/cord/provenance.json` 的原始/可用数量、标注映射和精确重复分组。少量多值字段与本题单值 schema 不兼容，被记录到 `excluded.jsonl`，不擅自选一个值作为真值；下方 manifests 的数量因此少于官方 split。所有质量结论仅适用于可用子集，必须报告覆盖率。

当前已实际下载并处理全部 1,000 张：可用 train **782/800**、validation **99/100**、test **98/100**，共隔离 21 条。数据在当前 workspace 的 `data/cord/`；Colab 中重新运行下载单元格。

```text
data/cord/
  train.jsonl       # 官方 800 中可表示为本题 schema 的样本
  validation.jsonl  # 官方 100 中可用的样本，模型选择/调参
  test.jsonl        # 官方 100 中可用的样本，方案冻结后评估
  images/<split>/...
  provenance.json
```

标注投影和抽取范围写在 BRIEF 中，必须先人工核对。公开数据集可能出现在模型预训练中，不可把结果当作未知客户的泛化保证。

## 先跑 baseline

先确认单张图片能够生成，再跑完整 validation：

```bash
python -m receipt_lab.predict --manifest data/cord/validation.jsonl --limit 2 --output runs/baseline-smoke.jsonl
python -m receipt_lab.evaluate --manifest data/cord/validation.jsonl --limit 2 --predictions runs/baseline-smoke.jsonl --output runs/baseline-smoke.metrics.json

python -m receipt_lab.predict --manifest data/cord/validation.jsonl --output runs/baseline.jsonl
python -m receipt_lab.evaluate --manifest data/cord/validation.jsonl --predictions runs/baseline.jsonl --output runs/baseline.metrics.json
python -m receipt_lab.report --baseline runs/baseline.jsonl --output runs/baseline-review.html
```

预测文件保存原始输出、schema/JSON 状态、截断标记、token 数和单次处理时间。不要编辑预测文件来清理失败样本。OOM 会直接停止；评估完整 manifest 时缺失输出会计失败。

## 真实 LoRA 训练：先 2 steps，再正式实验

```bash
python -m receipt_lab.train --limit 8 --max-steps 2 --output runs/gpu-smoke
python -m receipt_lab.predict --adapter runs/gpu-smoke --limit 2 --output runs/gpu-smoke-reloaded.jsonl

python -m receipt_lab.train --output runs/lora-v1
python -m receipt_lab.predict --adapter runs/lora-v1 --output runs/lora-v1.validation.jsonl
python -m receipt_lab.evaluate --predictions runs/lora-v1.validation.jsonl --output runs/lora-v1.metrics.json
python -m receipt_lab.report --baseline runs/baseline.jsonl --candidate runs/lora-v1.validation.jsonl --output runs/comparison.html
```

输出目录非空时拒绝覆盖，用新的 run 名保留每次实验。默认配置的 100 optimizer steps × gradient accumulation 8 × batch size 1 大约是一遍 train 数据，训练时间必须实测，不能承诺某张 Colab 卡一定在一小时内完成。

训练只监督 assistant JSON 和结束 token；prompt、图片占位 token 与 padding 不计入 loss。Qwen 模型内部负责 causal shift，collator 不提前 shift。

默认 LoRA 目标为语言注意力 q/k/v/o projections。微调后仅保存 adapter，同时保存 processor、模型资源配置、Trainer state 和 train IDs。推理加载相同固定 revision 的基础模型再加载 adapter，不依赖内存中尚未保存的训练实例。

图像 token 预算、学习率、步数、rank 等配置分别在 `configs/baseline.json` 和 `configs/train_t4.json`。做实验请复制配置，每次只改变明确的因素。超过 token 预算时明确报错，不静默截断答案或图像 token。

## 冻结方案后做 final test

```bash
python -m receipt_lab.predict --manifest data/cord/test.jsonl --output runs/baseline.test.jsonl
python -m receipt_lab.predict --manifest data/cord/test.jsonl --adapter runs/lora-v1 --output runs/lora-v1.test.jsonl
python -m receipt_lab.evaluate --manifest data/cord/test.jsonl --predictions runs/baseline.test.jsonl --output runs/baseline.test.metrics.json
python -m receipt_lab.evaluate --manifest data/cord/test.jsonl --predictions runs/lora-v1.test.jsonl --output runs/lora-v1.test.metrics.json
```

## 推理集成演示

```bash
python -m receipt_lab.serve --adapter runs/lora-v1
```

另一个终端用本地请求验证，例如 `curl -F file=@data/cord/images/validation/validation-0000.png http://127.0.0.1:8000/extract`。默认绑定 loopback，仅作本地集成演示；云 onboarding 不提供 localhost 浏览器预览。

单 worker 加锁避免多个线程同时占用 GPU；提供图片大小检查和结构错误状态。未实现鉴权、请求总超时、队列、批处理、监控、租户隔离和生产容量测试，这些是需要你在 review 说明的局限。GPU OOM 等基础设施故障不应被包装成成功输出。

## 当前 CPU 环境的 smoke 验证

当前 workspace 已准备好独立 CPU 环境和真实数据，可以直接使用：

```bash
source /workspace/reducto-onsite-runtime/venv/bin/activate
export HF_HOME=/workspace/reducto-onsite-runtime/hf-cache
export TOKENIZERS_PARALLELISM=false
python -m receipt_lab.train --smoke-model --limit 2 --max-steps 2 --output runs/cpu-check
python -m receipt_lab.predict --adapter runs/cpu-check --limit 2 --output runs/cpu-check.predictions.jsonl
python -m receipt_lab.evaluate --limit 2 --predictions runs/cpu-check.predictions.jsonl --output runs/cpu-check.metrics.json
```

此模式使用真实 Qwen processor、真实 CORD 样本和缩小的随机 Qwen 多模态架构，通过真实 TRL/PEFT 执行更新、保存和重载。它只验证实现链路，不能证明预训练 3B 模型质量、GPU 量化兼容性或显存预算。tiny-model 的错误输出不代表 baseline 质量。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| BRIEF.md | 候选人题目与验收标准 |
| INTERVIEW_EVENTS.md | 按阶段打开的追加需求 |
| SOURCES.md / upstream/ | 模型、数据、官方代码与许可来源 |
| receipt_lab/prepare.py | 数据下载、固定 split、标注映射 |
| receipt_lab/modeling.py | Qwen 加载、精度/量化、answer-only collator |
| receipt_lab/train.py | TRL + PEFT 训练、adapter 保存 |
| receipt_lab/predict.py | baseline / adapter 批量推理 |
| receipt_lab/evaluate.py | 自定义评分、失败分母、切片与延迟 |
| receipt_lab/report.py | 原图、答案、两个版本预测的静态浏览页 |
| receipt_lab/serve.py | 本地上传图片推理接口 |
| EXPERIMENT_LOG.md | 留给你填写的实验与决策记录 |
| VALIDATION.md | 当前实例已验证与尚未验证的边界 |

数据、缓存、权重和 runs 均被 Git 忽略。临时 CPU 环境在 `/workspace/reducto-onsite-runtime/venv`；Colab 不依赖这个路径。
