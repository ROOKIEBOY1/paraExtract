# paraExtract

基于 PaddleNLP PP-UIE 的本地中文动态 Schema 信息抽取实验，当前聚焦于 DJI Pocket 参数文本。项目提供 PP-UIE-0.5B/1.5B 的本地下载、离线加载、命令行推理、严格结果校验、性能评估和可视化测试页面。

> 这是验证性实验，不包含微调、服务化部署、量化或端侧转换。模型权重不会提交到仓库，需要首次运行时单独下载。

## 功能

- 支持 `paddlenlp/PP-UIE-0.5B` 和 `paddlenlp/PP-UIE-1.5B`；
- 模型、Tokenizer、配置文件保存到明确的本地目录；
- 支持运行时动态切换 Schema，无需重新加载当前模型；
- 提供 Web 页面选择文本、字段、模型和严格抽取模式；
- 0.5B/1.5B 可从页面切换，切换时先释放当前模型，再加载目标模型；
- 保留模型原始输出，同时输出标准化结果、原文证据和拒绝原因；
- 包含 6 段人工整理的 Pocket 参数测试文本和自动化测试。

## 已验证环境

| 组件 | 版本/配置 |
| --- | --- |
| 操作系统 | macOS arm64 |
| Python | 3.9 |
| PaddlePaddle | 3.3.0 |
| PaddleNLP | 3.0.0b4 |
| 推理设备 | CPU |
| 推理精度 | FP32 |

Apple GPU 不是 Paddle CUDA GPU，因此当前 macOS 实验使用 CPU。完整环境探测结果见 [`results/environment.md`](results/environment.md)。

## 快速开始

### 1. 创建独立环境

Conda：

```bash
conda create -n paraextract python=3.9 -y
conda activate paraextract
python -m pip install -r requirements.txt
```

如果没有 Conda，可使用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 下载并检查模型

```bash
python scripts/download_models.py --model all
python scripts/download_models.py --model all --verify-only
```

模型默认保存在 `models/PP-UIE-0.5B/` 和 `models/PP-UIE-1.5B/`。该目录已被 `.gitignore` 排除。

### 3. 启动可视化页面

```bash
python scripts/web_ui.py --model 0.5b --port 7860 --max-length 1024
```

打开 <http://127.0.0.1:7860/>。页面支持：

- 选择内置测试文本或直接粘贴文本；
- 选择需要抽取的字段；
- 切换 0.5B/1.5B；
- 开启或关闭严格抽取模式；
- 查看标准化结果、原文证据、模型原始输出和拒绝原因；
- 查看不包含模型加载时间的 Schema 设置与推理耗时。

切换模型时只保留当前选中的模型对象。Paddle CPU 分配器可能缓存已经释放的空闲内存页，因此系统监视器中的进程 RSS 不一定立即下降，但旧模型权重引用已经移除。

## 命令行推理

```bash
python scripts/infer.py \
  --model 0.5b \
  --schema configs/pocket_schema.json \
  --text-file testdata/pocket_user_six_samples.jsonl \
  --output results/outputs_0.5b.jsonl \
  --device cpu \
  --precision float32 \
  --max-length 1024
```

将 `--model` 改为 `1.5b` 即可测试 1.5B。直接输入文本时使用 `--text` 代替 `--text-file`；加上 `--offline` 可阻止推理阶段访问网络。

## 严格抽取模式

模型原始输出始终保留在 `raw_output`。严格模式不会修改模型答案，而是要求候选同时满足：

1. 值在原文中逐字出现；
2. 候选所在位置具有对应字段证据，或符合无歧义格式；
3. 标签后的直接取值与候选一致；
4. 候选通过字段类型和允许值校验。

通过校验的结果写入 `verified_output`；不合格候选写入 `rejected_candidates`：

| 原因 | 含义 |
| --- | --- |
| `value_not_in_source` | 模型答案没有在原文中出现 |
| `value_not_allowed` | 值不符合字段类型或允许范围 |
| `field_evidence_missing` | 找到了相同文本，但缺少对应字段上下文 |
| `field_value_mismatch` | 候选不是字段标签后的直接取值 |
| `numeric_sign_mismatch` | 模型遗漏了原文中的正负号 |

当前严格字段目录定义在 [`pp_uie/strict_validation.py`](pp_uie/strict_validation.py)。英文名称可以映射到标准中文 Schema，但原文标签标准化仍属于后续改进项。

## 测试与评估

```bash
python -m pytest -q
```

服务启动后可复测 6 段 Pocket 文本：

```bash
python scripts/evaluate_strict_ui.py
```

严格模式评估结论见 [`results/strict_mode_assessment.md`](results/strict_mode_assessment.md)，结构化数据见 [`results/strict_mode_evaluation.json`](results/strict_mode_evaluation.json)。

## 项目结构

```text
.
├── configs/                 # Schema 配置
├── pp_uie/                  # 模型后端、校验、指标和服务逻辑
├── scripts/                 # 环境检查、下载、推理、评估和 Web 入口
├── testdata/                # 人工转录文本、期望结果和噪声样例
├── tests/                   # 自动化测试
├── web/                     # 本地可视化页面
├── results/                 # 可公开的环境与评估报告
├── requirements.txt
└── README.md
```

## 已知限制

- macOS 当前只能使用 Paddle CPU 推理，速度明显慢于兼容的 CUDA GPU；
- 1.5B 在 CPU 上的加载和推理成本显著高于 0.5B；
- 字段较多、文本较长时，生成式模型可能返回字段标签、错误关联或不存在字段的候选；
- 严格模式可以提高精度，但不能恢复模型完全漏掉的正确值；
- 当前字段目录以 Pocket 参数为主，未来计划改为配置驱动的字段、别名和类型注册表。

## 数据与模型

- 测试文本为本项目人工整理或构造的数据；
- 图片样例仅用于验证人工转录流程；
- PP-UIE 模型文件来源于 PaddleNLP，使用时需遵守对应许可和条款；
- 请勿提交本地模型权重、缓存、私人数据或包含敏感信息的推理结果。
