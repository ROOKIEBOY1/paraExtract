# UIE-mini Pocket 参数评估

- 样本数：46
- 模型文件大小：108000442 bytes
- 模型加载时间：2823.291 ms
- 离线环境运行：成功
- 通过严格校验的候选：129
- 被拒绝候选：160

## 推理时延（不含加载）

- 平均：2145.554 ms
- P50：2026.525 ms
- P95：2460.318 ms
- 最小 / 最大：2009.064 / 5128.860 ms

## 代表性拒绝案例

```json
[
  {
    "sample_id": "pocket_user_02",
    "field": "纹理",
    "value": "ISO",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "ISO：50-800"
  },
  {
    "sample_id": "pocket_user_04",
    "field": "纹理",
    "value": "0去噪",
    "reasons": [
      "value_not_allowed"
    ],
    "evidence": "图像调节：纹理0去噪0"
  },
  {
    "sample_id": "pocket_user_04",
    "field": "色彩模式",
    "value": "普通\n\n对焦",
    "reasons": [
      "value_not_in_source"
    ]
  },
  {
    "sample_id": "pocket_user_05",
    "field": "白平衡",
    "value": "自动\n▪️ISO：800",
    "reasons": [
      "value_not_in_source",
      "value_not_allowed"
    ]
  },
  {
    "sample_id": "pocket_user_05",
    "field": "锐度",
    "value": "+10",
    "reasons": [
      "value_not_allowed"
    ],
    "evidence": "▫️锐化：+10\\"
  },
  {
    "sample_id": "pocket_user_05",
    "field": "纹理",
    "value": "荔枝",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "✔️绿巨能action街拍套装：兔笼+街拍手柄的组合，兔笼是全包式铝合金，给相机整体性的框架保护，加上这个手柄有一种按单反的顺滑感，手柄处还是荔枝纹的，非常高级有质感，刚看到图片就被颜值俘获了😍"
  },
  {
    "sample_id": "pocket_user_05",
    "field": "滤镜浓度",
    "value": "NC70%\n\n🎨",
    "reasons": [
      "value_not_in_source"
    ]
  },
  {
    "sample_id": "pocket_user_05",
    "field": "滤镜浓度",
    "value": "4k50",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "▪️分辨率：4k50"
  },
  {
    "sample_id": "pocket_user_05",
    "field": "感光度",
    "value": "4k50",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "▪️分辨率：4k50"
  },
  {
    "sample_id": "pocket_user_05",
    "field": "快门速度",
    "value": "4k50",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "▪️分辨率：4k50"
  }
]
```

原始跨度、概率和偏移量完整保存在 `uie_mini_outputs.jsonl` 的 `raw_output` 中；本报告不改写模型预测。
