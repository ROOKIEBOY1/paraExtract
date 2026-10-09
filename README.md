# paraExtract

基于 PaddleNLP 的本地中文动态 Schema 信息抽取实验，当前聚焦于 DJI Pocket 参数文本。项目提供 PP-UIE-0.5B、PP-UIE-1.5B、UIE-mini 和 UIE-base 的本地下载、离线加载、严格结果校验、性能评估和可视化测试页面。

> 这是验证性实验，不包含微调、服务化部署、量化或端侧转换。模型权重不会提交到仓库，需要首次运行时单独下载。

## 功能

- 支持生成式 `paddlenlp/PP-UIE-0.5B`、`paddlenlp/PP-UIE-1.5B` 和跨度式 `uie-mini`、`uie-base`；
- 模型、Tokenizer、配置文件保存到明确的本地目录；
- 支持运行时动态切换 Schema，无需重新加载当前模型；
- 提供 Web 页面选择文本、字段、模型和严格抽取模式；
- 四个模型可从页面切换，切换时先释放当前模型，再加载目标模型；
- 保留模型原始输出，同时输出标准化结果、原文证据和拒绝原因；
- 包含 6 段人工整理的 Pocket 参数文本、40 条边界/噪声测试文本和自动化数据契约测试。

## 模型架构

项目使用统一的 `text_extractor` Python 包，但将两类模型实现保持独立：

- `text_extractor.backends.pp_uie`：PP-UIE-0.5B/1.5B 生成式后端，负责提示词构造、生成结果解析以及对应权重的下载和校验；
- `text_extractor.backends.uie`：UIE-mini/base 跨度抽取后端，负责 Taskflow 推理、原文偏移和动态 Schema 别名，以及对应权重的下载和校验；
- `text_extractor.core`、`validation`、`evaluation`、`runtime` 和 `web`：两类模型共同使用的 Schema、严格校验、评测、运行时与页面服务。

开发者可以从各自后端导入模型接口：

```python
from text_extractor.backends.pp_uie import PPUIEBackend, RuntimeConfig
from text_extractor.backends.uie import UIETaskflowBackend, UIETaskflowConfig
```

两个后端不会互相导入或加载；页面切换模型时，内存中仍只保留当前选择的一个模型。

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

模型默认保存在：

- `models/PP-UIE-0.5B/`
- `models/PP-UIE-1.5B/`
- `models/UIE-mini/`
- `models/UIE-base/`

`models/UIE-mini/` 与 `models/UIE-base/` 首次下载的源文件必须包括：

- `model_state.pdparams`
- `config.json`
- `vocab.txt`
- `special_tokens_map.json`
- `tokenizer_config.json`

下载器还会写入包含 URL、字节数、官方 MD5 和本地 SHA-256 的 `download_manifest.json`。UIE-mini/UIE-base 第一次加载时，Taskflow 会在各自目录生成 `static/inference.json`、`static/inference.pdiparams` 和 `.cache_info`；这些是可重建的本地静态推理产物。整个 `models/` 目录已被 `.gitignore` 排除，不会提交权重或派生文件。

只下载 UIE-mini 时使用：

```bash
python scripts/download_models.py --model uie-mini --destination models
python scripts/download_models.py --model uie-mini --destination models --verify-only
```

只下载 UIE-base 时使用：

```bash
python scripts/download_models.py --model uie-base --destination models
python scripts/download_models.py --model uie-base --destination models --verify-only
```

### 3. 启动可视化页面

```bash
python scripts/web_ui.py --model 0.5b --port 7860 --max-length 1024
```

直接以 UIE-mini 启动：

```bash
python scripts/web_ui.py --model uie-mini --port 7860 --max-length 512
```

以 UIE-base 启动：

```bash
python scripts/web_ui.py --model uie-base --port 7860 --max-length 512
```

打开 <http://127.0.0.1:7860/>。页面支持：

- 选择内置测试文本或直接粘贴文本；
- 下拉框按原有6条、P0、P1和综合压力场景分组提供全部46条测试文本；
- 选择需要抽取的字段；
- 切换 PP-UIE-0.5B、PP-UIE-1.5B、UIE-mini 或 UIE-base；
- 开启或关闭严格抽取模式；
- 查看标准化结果、原文证据、模型原始输出和拒绝原因；
- 查看不包含模型加载时间的 Schema 设置与推理耗时。

切换模型时先关闭并移除当前后端、执行内存回收，再构造目标后端，因此 Python 中只保留一个活动模型对象。如果目标模型加载失败，服务会明确进入“无活动模型”状态，页面允许重新选择，不会假装旧模型仍在运行。Paddle CPU 分配器可能缓存已经释放的空闲内存页，因此系统监视器中的进程 RSS 不一定立即下降；判断是否共存应以活动后端引用和切换前后的增量为准，而不是要求 RSS 立即回到进程初始值。

PP-UIE-0.5B/1.5B 是生成式抽取：模型按每个 Schema 生成答案文本。UIE-mini/UIE-base 使用官方 `Taskflow("information_extraction", model=...)` 的跨度抽取，固定 `position_prob=0.5`，其 `raw_output` 原样保留 `text`、`start`、`end` 和 `probability`。当前阶段不提供阈值滑块，也不使用概率绕过严格证据校验。

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
| `span_offset_mismatch` | UIE span 的 `start/end` 与原文切片不一致 |

当前严格字段目录和英文别名定义在 [`text_extractor/validation/strict.py`](text_extractor/validation/strict.py)。UIE-mini/UIE-base 每次抽取前会检查原文：只有原文实际出现 `ISO`、`White Balance`、`Sharpness` 等已登记英文标签时，才在本次 Taskflow Schema 中追加对应的“别名值”查询；模型输出随后合并回标准中文字段。原文不会被翻译或改写，因此 `start`、`end` 和证据片段仍对应用户提交的原始文本。未命中的别名不会加入 Schema，PP-UIE-0.5B/1.5B 的生成式推理路径也不受影响。新增字段时只需在字段配置中登记别名，无需修改这段调度逻辑；别名查询会增加本次 UIE 的查询项，英文标签较多时推理耗时可能相应上升。

## 测试与评估

```bash
python -m pytest -q
```

服务启动后可复测 6 段 Pocket 文本：

```bash
python scripts/evaluate_strict_ui.py
```

严格模式评估结论见 [`results/strict_mode_assessment.md`](results/strict_mode_assessment.md)，结构化数据见 [`results/strict_mode_evaluation.json`](results/strict_mode_evaluation.json)。

### UIE-mini 的 46 条评估与离线复测

```bash
python scripts/evaluate_uie_mini.py \
  --model-path models/UIE-mini \
  --output results/uie_mini_outputs.jsonl \
  --report results/uie_mini_assessment.md
```

离线复测至少设置以下变量；评估器会在报告中记录是否处于该离线配置：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AISTUDIO_OFFLINE=1 \
python scripts/evaluate_uie_mini.py \
  --model-path models/UIE-mini \
  --limit 1 \
  --output results/uie_mini_offline_smoke.jsonl \
  --report results/uie_mini_offline_smoke.md
```

本机还用阻断 socket、`requests` 和 `urllib` 的验证进程完成了相同抽取，确认静态文件生成后不再请求远端。正式结果见 [`results/uie_mini_outputs.jsonl`](results/uie_mini_outputs.jsonl)，统计和错误案例见 [`results/uie_mini_assessment.md`](results/uie_mini_assessment.md)。本次 CPU 实测：模型源文件约 108 MB，离线加载约 2.82 秒，46 条平均推理约 2.15 秒，P50 约 2.03 秒，P95 约 2.46 秒；耗时会随 CPU、文本长度和 Schema 字段数变化。

UIE-base 使用同一评估入口，仅增加 `--model uie-base`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AISTUDIO_OFFLINE=1 \
python scripts/evaluate_uie_mini.py \
  --model uie-base \
  --model-path models/UIE-base \
  --output results/uie_base_outputs.jsonl \
  --report results/uie_base_assessment.md
```

| 模型 | 源文件大小 | 离线加载 | 平均推理 | P50 | P95 | 通过候选 | 拒绝候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UIE-mini | 108.0 MB | 2.82 s | 2.15 s | 2.03 s | 2.46 s | 129 | 160 |
| UIE-base | 472.0 MB | 3.19 s | 5.85 s | 5.42 s | 6.22 s | 300 | 32 |

UIE-base 正式结果见 [`results/uie_base_outputs.jsonl`](results/uie_base_outputs.jsonl) 和 [`results/uie_base_assessment.md`](results/uie_base_assessment.md)。两者使用相同 46 条文本、Schema、`position_prob=0.5` 与严格校验器；“通过候选更多”反映本测试集上的有效召回和边界质量提升，但不等同于人工金标准确率。

UIE-base 页面端到端验收中，新进程加载后的 RSS 为 1,492,960 KiB；UIE-base → UIE-mini → UIE-base 均切换成功，后两次加载分别约 0.36 秒和 1.31 秒。切换到 UIE-mini 后 RSS 基本不变（1,492,928 KiB），这是 Paddle 分配器复用已申请内存页的表现；服务注册表中的活动后端仍只有当前模型一个。

端到端页面验收中，`/api/config` 返回四个带 `id/label` 的模型项，UIE-mini 抽取保留了完整跨度元数据，UIE-mini → 0.5B → UIE-mini 均切换成功。新进程只加载 UIE-mini 时 RSS 为 966,752 KiB；经历 1.5B 后 RSS 为 10,190,880 KiB，切回 UIE-mini 后降为 3,746,480 KiB；随后 0.5B 为 5,552,016 KiB，再切回 UIE-mini 为 5,438,768 KiB。后两者高于新进程基线，印证了 Paddle CPU 分配器保留空闲页，但活动服务与后端引用始终只有一个，RSS 也没有保持为两个模型峰值之和。

### Pocket 边界测试集

新增测试集以现有 6 条 Pocket 文本为母本，覆盖字段别名、缺省值、同名歧义、互斥参数、范围/候选值、非法参数、机内/后期分组以及 OCR 噪声：

- 输入：[`testdata/pocket_edge_cases.jsonl`](testdata/pocket_edge_cases.jsonl)
- 人工金标：[`testdata/pocket_edge_cases_expected.jsonl`](testdata/pocket_edge_cases_expected.jsonl)
- 覆盖矩阵与输出规范：[`testdata/pocket_edge_cases_matrix.md`](testdata/pocket_edge_cases_matrix.md)

测试文本和金标刻意分开保存。缺失字段不输出空占位；连续范围保留 `起点-终点`，离散多候选使用 `value|value`；无法消歧、互斥或非法候选不会进入标准化 `params`。

40 条边界文本均按真实参数推荐帖重写：每条 Schema 覆盖 11 个字段，至少包含 7 个明确可评测参数，再叠加一个受控的歧义、缺省、冲突、非法值或 OCR 噪声场景。

## 项目结构

```text
.
├── configs/                 # Schema 配置
├── text_extractor/          # 统一的文本抽取 Python 包
│   ├── backends/
│   │   ├── pp_uie/          # PP-UIE-0.5B/1.5B 生成式后端与模型管理
│   │   └── uie/             # UIE-mini/base 跨度后端与模型管理
│   ├── core/                # Schema、注册表、标准化和操作日志
│   ├── validation/          # 证据、字段类型和值域校验
│   ├── evaluation/          # 指标、基准测试和资源监控
│   ├── runtime/             # CLI、环境检查和离线控制
│   └── web/                 # 常驻模型服务与本地 HTTP 接口
├── scripts/                 # 环境检查、下载、推理、评估和 Web 入口
├── testdata/                # 人工转录文本、期望结果和噪声样例
├── tests/                   # 自动化测试
├── web/                     # 本地可视化页面
├── results/                 # 可公开的环境与评估报告
├── requirements.txt
└── README.md
```

旧的 `pp_uie` Python 包已经移除，不提供双路径兼容层。外部运行命令、模型 ID、`models/` 下的权重位置、Web API 和结果格式保持不变；仅开发者代码中的 Python 导入路径迁移到 `text_extractor.*`。

## 已知限制

- macOS 当前只能使用 Paddle CPU 推理，速度明显慢于兼容的 CUDA GPU；
- 1.5B 在 CPU 上的加载和推理成本显著高于 0.5B；
- UIE-mini 的跨度输出更可追溯，但在 Pocket 参数帖子上仍会出现跨行边界、字段串扰和把后期参数当作机内参数的情况；本次 46 条中严格模式通过 129 个候选、拒绝 160 个候选；
- UIE-base 在相同 46 条上明显减少了错误边界和字段串扰，但约 4.4 倍的源权重、约 2.7 倍的平均 CPU 时延使其更适合效果优先而非低资源场景；
- 字段较多、文本较长时，生成式模型可能返回字段标签、错误关联或不存在字段的候选；
- 严格模式可以提高精度，但不能恢复模型完全漏掉的正确值；
- PaddleNLP 3.0.0b4 写入的 Taskflow `.cache_info` 使用旧的 `taskflow` 标记，而 Paddle 3.3.0 静态模型后缀为 `.json`，会导致重复静态转换；本项目只在完整静态文件存在且 marker 对应当前权重 MD5 时修正该 marker，不修改安装包；
- Paddle 3.3.0 的首次 `to_static` 转换会临时使用其硬编码的 `~/.cache/paddle/to_static_tmp`，而最终静态模型仍分别保存在 `models/UIE-mini/static/` 和 `models/UIE-base/static/`；受限沙箱需要为首次转换允许该临时缓存写入，普通本地环境通常无需额外处理；
- 当前字段目录以 Pocket 参数为主，未来计划改为配置驱动的字段、别名和类型注册表。

## 数据与模型

- 测试文本为本项目人工整理或构造的数据；
- 图片样例仅用于验证人工转录流程；
- PP-UIE、UIE-mini 与 UIE-base 模型文件来源于 PaddleNLP，使用时需遵守对应许可和条款；
- 请勿提交本地模型权重、缓存、私人数据或包含敏感信息的推理结果。
