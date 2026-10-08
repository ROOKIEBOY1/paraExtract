from pp_uie.backend import GenerationEngine, PaddleGenerationEngine, PPUIEBackend, RuntimeConfig, _parse_answer


class FixtureEngine(GenerationEngine):
    def generate(self, prompts, max_new_tokens):
        return [[{"text": "夜景赛博朋克"}] for _ in prompts]


def test_schema_change_does_not_reload_model(tmp_path):
    loads = []
    backend = PPUIEBackend(RuntimeConfig(tmp_path, "cpu", "float32", 1, 512, 50),
                           engine_factory=lambda config: loads.append(config) or FixtureEngine())
    backend.load()
    backend.set_schema(["拍照公式场景"])
    backend.extract(["夜景赛博朋克\n曝光:+0.7EV"])
    backend.set_schema({"拍照公式场景": ["曝光"]})
    backend.extract(["夜景赛博朋克\n曝光:+0.7EV"])
    assert len(loads) == 1


def test_extract_batches_top_level_prompts_across_texts(tmp_path):
    calls = []

    class RecordingEngine(GenerationEngine):
        def generate(self, prompts, max_new_tokens):
            calls.append(list(prompts))
            return [[{"text": f"scene-{index}"}] for index, _ in enumerate(prompts)]

    backend = PPUIEBackend(
        RuntimeConfig(tmp_path, "cpu", "float32", 2, 512, 50),
        engine_factory=lambda config: RecordingEngine(),
    )
    backend.load()
    backend.set_schema(["拍照公式场景"])
    output = backend.extract(["text-a", "text-b", "text-c"])

    assert [len(call) for call in calls] == [2, 1]
    assert [row["拍照公式场景"][0]["text"] for row in output] == [
        "scene-0",
        "scene-1",
        "scene-0",
    ]


def test_paddle_generate_output_is_decoded_without_prompt_length_slice():
    generated_only = object()

    class InputIds:
        shape = (1, 91)

    class Tokenizer:
        chat_template = None
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            return {"input_ids": InputIds()}

        def batch_decode(self, sequences, skip_special_tokens):
            assert sequences is generated_only
            return ["夜景赛博朋克"]

    class Model:
        def generate(self, **kwargs):
            return generated_only

    engine = object.__new__(PaddleGenerationEngine)
    engine.tokenizer = Tokenizer()
    engine.model = Model()
    engine.generation_config = object()
    engine.config = RuntimeConfig(__import__("pathlib").Path("."))
    assert engine.generate(["prompt"], 50) == [[{"text": "夜景赛博朋克"}]]


def test_answer_parser_stops_at_official_end_marker():
    answer = "夜景赛博朋克\n**回答结束**\n请提供更多的信息"
    assert _parse_answer(answer) == [{"text": "夜景赛博朋克"}]


def test_answer_parser_preserves_negative_sign():
    assert _parse_answer("-0.3EV") == [{"text": "-0.3EV"}]


def test_engine_applies_embedded_chat_template_before_tokenization():
    seen = {}

    class Tokenizer:
        chat_template = "fixture"
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(self, text, tokenize=False):
            assert tokenize is False
            return f"CHAT:{text}"

        def __call__(self, texts, **kwargs):
            seen["texts"] = texts
            seen["kwargs"] = kwargs
            return {"input_ids": object()}

        def batch_decode(self, sequences, skip_special_tokens):
            return ["实体"]

    class Model:
        def generate(self, **kwargs):
            seen["generate"] = kwargs
            return object()

    engine = object.__new__(PaddleGenerationEngine)
    engine.tokenizer = Tokenizer()
    engine.model = Model()
    engine.generation_config = object()
    engine.config = RuntimeConfig(__import__("pathlib").Path("."))
    engine.generate(["PROMPT"], 50)
    assert seen["texts"] == ["CHAT:PROMPT"]
    assert seen["kwargs"]["truncation_side"] == "left"
    assert seen["generate"]["eos_token_id"] == 2
