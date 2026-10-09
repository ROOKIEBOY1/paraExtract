"""Local-path PP-UIE generation backend with dynamic schemas."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

from ...core.schema import SchemaNode, parse_schema


LLM_IE_PROMPT = """你是一个阅读理解专家，请提取所给句子与问题，提取实体。请注意，如果存在实体，则一定在原句中逐字出现，请输出对应实体的原文，不要进行额外修改；如果无法提取，请输出“无相应实体”。
**句子开始**
{sentence}
**句子结束**
**问题开始**
{prompt}
**问题结束**
**回答开始**
"""


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path
    device: str = "cpu"
    precision: str = "float32"
    batch_size: int = 1
    max_length: int = 512
    max_new_tokens: int = 50


class GenerationEngine:
    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[list[dict[str, str]]]:
        raise NotImplementedError


def _parse_answer(answer: str) -> list[dict[str, str]]:
    answer = answer.split("**回答结束**", 1)[0].strip()
    if not answer or "无相应实体" in answer:
        return []
    try:
        value = json.loads(answer)
        if isinstance(value, list):
            return [{"text": str(item.get("text", "")) if isinstance(item, dict) else str(item)} for item in value]
    except json.JSONDecodeError:
        pass
    # A hyphen can be a list marker, but it is also the sign of a legitimate
    # extracted value such as ``-0.3EV``.  Only remove unambiguous bullet
    # characters here; validation downstream handles any remaining prose.
    values = [part.strip(" \t•，,。\"'") for part in re.split(r"[\n、]+", answer)]
    return [{"text": value} for value in values if value]


class PaddleGenerationEngine(GenerationEngine):
    def __init__(self, config: RuntimeConfig):
        os.environ.setdefault("PPNLP_HOME", str(config.model_path.resolve().parent / ".paddlenlp-cache"))
        import paddle
        from paddlenlp.generation import GenerationConfig
        from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

        paddle.set_device(config.device)
        local_path = str(config.model_path.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, padding_side="left")
        self.generation_config = GenerationConfig.from_pretrained(local_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path, dtype=config.precision, use_flash_attention=False
        )
        self.model.eval()
        self.config = config

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[list[dict[str, str]]]:
        model_inputs = list(prompts)
        if self.tokenizer.chat_template is not None:
            model_inputs = [self.tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in model_inputs]
        encoded = self.tokenizer(
            model_inputs,
            return_tensors="pd",
            return_position_ids=True,
            padding_side="left",
            padding=True,
            max_new_tokens=self.config.max_length,
            truncation=True,
            truncation_side="left",
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        generated = self.model.generate(
            **encoded,
            decode_strategy="greedy_search",
            max_new_tokens=max_new_tokens,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
            use_cache=True,
        )
        sequences = generated[0] if isinstance(generated, tuple) else generated
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [_parse_answer(answer) for answer in decoded]


class PPUIEBackend:
    def __init__(self, config: RuntimeConfig, engine_factory: Callable[[RuntimeConfig], GenerationEngine] = PaddleGenerationEngine):
        self.config = config
        self.engine_factory = engine_factory
        self.engine: GenerationEngine | None = None
        self.schema: tuple[SchemaNode, ...] = ()

    def load(self) -> None:
        if self.engine is None:
            self.engine = self.engine_factory(self.config)

    def close(self) -> None:
        """Drop model/tokenizer references so the inactive model can be reclaimed."""
        self.engine = None

    def set_schema(self, schema: Any) -> None:
        self.schema = parse_schema(schema)

    def _ask(self, text: str, prompt: str) -> list[dict[str, Any]]:
        return self._ask_many([(text, prompt)])[0]

    def _ask_many(self, questions: Sequence[tuple[str, str]]) -> list[list[dict[str, Any]]]:
        assert self.engine is not None
        answers: list[list[dict[str, Any]]] = []
        for start in range(0, len(questions), self.config.batch_size):
            chunk = questions[start:start + self.config.batch_size]
            prompts = [LLM_IE_PROMPT.format(sentence=text, prompt=prompt) for text, prompt in chunk]
            generated = self.engine.generate(prompts, self.config.max_new_tokens)
            answers.extend([[dict(item) for item in values] for values in generated])
        return answers

    def _fill_children(self, text: str, parent: dict[str, Any], children: tuple[SchemaNode, ...]) -> None:
        relations: dict[str, list[dict[str, Any]]] = {}
        for child in children:
            values = self._ask(text, f"{parent['text']}的{child.name}")
            if child.children:
                for value in values:
                    self._fill_children(text, value, child.children)
            if values:
                relations[child.name] = values
        if relations:
            parent["relations"] = relations

    def extract(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("backend is not loaded")
        if not self.schema:
            raise RuntimeError("schema is not set")
        outputs: list[dict[str, Any]] = [{} for _ in texts]
        for node in self.schema:
            answers = self._ask_many([(text, node.name) for text in texts])
            for text, record, values in zip(texts, outputs, answers):
                for value in values:
                    self._fill_children(text, value, node.children)
                record[node.name] = values
        return outputs
