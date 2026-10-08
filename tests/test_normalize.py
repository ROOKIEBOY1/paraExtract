from pp_uie.normalize import normalize_scene_output


def test_similar_concert_scenes_remain_separate():
    raw = {"拍照公式场景": [
        {"text": "演唱会拍摄舞台", "relations": {"分辨率": [{"text": "1.4K/60fps"}]}},
        {"text": "演唱会参数", "relations": {"分辨率": [{"text": "4K30p/60p"}]}},
    ]}
    result = normalize_scene_output(raw, {"拍照公式场景": ["分辨率"]})
    assert result["scenes"] == [
        {"scene": "演唱会拍摄舞台", "parameters": {"分辨率": ["1.4K/60fps"]}, "unlabeled_parameters": []},
        {"scene": "演唱会参数", "parameters": {"分辨率": ["4K30p/60p"]}, "unlabeled_parameters": []},
    ]


def test_normalization_preserves_raw_text_verbatim():
    raw = {"拍照公式场景": [{"text": "暗调氛围感", "relations": {"去噪": [{"text": "-0.1"}]}}]}
    result = normalize_scene_output(raw, {"拍照公式场景": ["去噪"]})
    assert result["scenes"][0]["parameters"]["去噪"] == ["-0.1"]

