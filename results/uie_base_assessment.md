# UIE-base Pocket 参数评估

- 样本数：46
- 模型文件大小：471995394 bytes
- 模型加载时间：3185.206 ms
- 离线环境运行：成功
- 通过严格校验的候选：300
- 被拒绝候选：32

## 推理时延（不含加载）

- 平均：5845.529 ms
- P50：5420.568 ms
- P95：6215.198 ms
- 最小 / 最大：4812.625 / 18808.584 ms

## 代表性拒绝案例

```json
[
  {
    "sample_id": "pocket_user_04",
    "field": "快门速度",
    "value": "ISO：50\\~3200",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "ISO：50\\~3200"
  },
  {
    "sample_id": "pocket_user_05",
    "field": "纹理",
    "value": "荔枝纹",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "✔️绿巨能action街拍套装：兔笼+街拍手柄的组合，兔笼是全包式铝合金，给相机整体性的框架保护，加上这个手柄有一种按单反的顺滑感，手柄处还是荔枝纹的，非常高级有质感，刚看到图片就被颜值俘获了😍"
  },
  {
    "sample_id": "pocket_edge_p0_01_01",
    "field": "曝光补偿",
    "value": "0.7EV",
    "reasons": [
      "value_not_allowed",
      "field_value_mismatch"
    ],
    "evidence": "曝光补偿+0.7EV"
  },
  {
    "sample_id": "pocket_edge_p0_01_01",
    "field": "快门速度",
    "value": "1/60s",
    "reasons": [
      "value_not_allowed"
    ],
    "evidence": "快门1/60s"
  },
  {
    "sample_id": "pocket_edge_p0_01_02",
    "field": "APP美颜",
    "value": "关闭\n锐度：0",
    "reasons": [
      "value_not_in_source",
      "value_not_allowed"
    ]
  },
  {
    "sample_id": "pocket_edge_p0_01_03",
    "field": "快门速度",
    "value": "1/120s\nEV-0.3",
    "reasons": [
      "value_not_in_source",
      "value_not_allowed"
    ]
  },
  {
    "sample_id": "pocket_edge_p0_01_03",
    "field": "锐度",
    "value": "去噪：1",
    "reasons": [
      "value_not_allowed",
      "field_evidence_missing"
    ],
    "evidence": "去噪：1"
  },
  {
    "sample_id": "pocket_edge_p0_02_01",
    "field": "快门速度",
    "value": "60",
    "reasons": [
      "value_not_allowed"
    ],
    "evidence": "快门：60"
  },
  {
    "sample_id": "pocket_edge_p0_02_02",
    "field": "曝光补偿",
    "value": "0.7EV",
    "reasons": [
      "value_not_allowed"
    ],
    "evidence": "曝光补偿0.7EV"
  },
  {
    "sample_id": "pocket_edge_p0_02_03",
    "field": "滤镜浓度",
    "value": "30",
    "reasons": [
      "value_not_allowed",
      "field_value_mismatch"
    ],
    "evidence": "滤镜浓度写的是30"
  }
]
```

原始跨度、概率和偏移量完整保存在 `uie_base_outputs.jsonl` 的 `raw_output` 中；本报告不改写模型预测。
