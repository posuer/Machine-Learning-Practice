# ML Interview 工作区（PyTorch）

此仓库用于长期准备 Senior Machine Learning Engineer 面试，以 PyTorch 为基础，围绕 training、modeling、eval、data 组织内容，不侧重推荐/搜索。当前只建立目录框架，不添加具体题目；每次由用户提供公司和岗位信息后再定制。默认采用 Python 3.12 + CPU PyTorch，不依赖 GPU、在线数据集或 API Key。

## 开始练习

```bash
cd /workspace/Machine-Learning-Practice
source /workspace/ml-interview-env/activate.sh
python /workspace/ml-interview-env/smoke.py
```

环境包含 PyTorch、torchvision、NumPy、SciPy、pandas、scikit-learn、Matplotlib、Seaborn、JupyterLab、pytest、Ruff、TensorBoard 和 tqdm。
环境依赖、版本快照和验证脚本在 `/workspace/ml-interview-env`；代码和笔记放在本仓库。云环境发布保存安装状态；长期积累的题解与笔记应单独提交到 Git。

## 目录框架

```text
python/
  training/       # 训练、优化、性能与调试
  modeling/       # 建模、模型结构与目标函数
  eval/           # 评估、实验与误差分析
  data/           # 数据流程、质量与一致性（代码/文档）
  fundamentals/   # 数学、经典 ML 与编程基础
  systems/        # 通用 ML 系统设计与生产化
notebooks/        # 后续交互式实验
tests/           # 后续实现的验证
companies/        # 按公司及面试批次归档
```

各目录目前只有职责说明，不包含题目、解答或预设公司内容。代码按以上主题归档；PyTorch 是实现工具，不另设重复的知识分类。原始数据、模型权重和训练日志放在仓库外。

## 每次练习的交付标准

1. 明确输入输出、shape、复杂度和边界条件。
2. 先完成可运行的简单实现，再解释向量化或性能改进。
3. 固定随机种子，避免训练/验证集泄漏，用小数据检查 loss 与梯度。
4. 能口述推导、设计取舍、失败情况，以及如何定位训练异常。
5. 记录用时、提示次数、错误原因和复习日期。

Senior MLE 的准备同时关注实现能力、实验可信度、生产化取舍和技术决策的解释。具体内容与优先级等收到公司、岗位、级别和轮次信息后再安排。大规模训练需要另行准备 GPU；当前环境支持小规模可验证实现。

## 公司准备模板

为每个公司复制以下栏目；公司的具体面试形式以当前招聘信息和本人确认的信息为准。

```markdown
# 公司 / 岗位 / 级别
- JD 或信息来源及日期：
- 面试时间与已确认轮次：
- 重点能力（与岗位证据对应）：
- 高频知识缺口：
- 本周练习（题目、验收标准、截止日期）：
- 模拟记录（用时、正确性、解释、提示）：
- 错题与下次复习日期：
- 项目经历：目标、个人贡献、指标、取舍、失败与改进
```

## 工具使用

激活环境后，针对已有题解运行 `pytest -q tests/` 和 `ruff check python/ tests/`。仓库目前还没有题解测试；环境 smoke 验证不等同于题库测试。

Notebook 可直接使用本环境的 Python kernel；命令行执行示例：

```bash
jupyter nbconvert --to notebook --execute notebooks/example.ipynb \
  --ExecutePreprocessor.kernel_name=ml-interview \
  --output-dir=/workspace/ml-interview-env/outputs
```

可选 JupyterLab 启动命令为 `jupyter lab --no-browser --ip=127.0.0.1 --port=8888`，保留默认认证；云 onboarding 界面不提供浏览器预览。TensorBoard 可按需使用 `tensorboard --logdir <训练日志目录>`。

## 环境维护

安装脚本为 `/workspace/ml-interview-env/install.sh`，依赖版本快照为该目录内的 `requirements.lock`。重新安装后运行 smoke 验证；更新依赖时应重新验证训练、torchvision 和 Notebook，并同步环境设置中的安装脚本。
使用已有 checkout；每个云任务已隔离，不需要创建 Git worktree。不要将虚拟环境、下载数据、模型权重或日志提交到 Git。
