from __future__ import annotations

import json

import httpx

from llm_stock_system.core.enums import Intent, StanceBias, TopicTag
from llm_stock_system.core.interfaces import QueryClassifier


_TIME_RANGE_LABELS = ["1d", "7d", "30d", "latest_quarter", "1y", "3y", "5y"]


def _build_schema() -> dict:
    """Derive the classification schema from enum source-of-truth.

    Wave 4 Stage 5: the LLM no longer emits ``question_type``. Routing is driven
    entirely by ``intent`` + ``topic_tags``; the InputLayer's rule engine
    internally generates the (transitional) question_type if Stage 6 still
    needs it. Keeps enum lists in sync with :class:`Intent` / :class:`TopicTag`
    automatically.
    """
    intents = [i.value for i in Intent]
    topic_tag_values = [t.value for t in TopicTag]
    stance_values = [s.value for s in StanceBias]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent",
            "topic_tags",
            "time_range_label",
            "stance_bias",
        ],
        "properties": {
            "intent": {"type": "string", "enum": intents},
            "topic_tags": {
                "type": "array",
                "items": {"type": "string", "enum": topic_tag_values},
            },
            "time_range_label": {"type": "string", "enum": _TIME_RANGE_LABELS},
            "stance_bias": {"type": "string", "enum": stance_values},
        },
    }


_SYSTEM_PROMPT = (
    "你是台股查詢分類器。根據使用者的自由文字問題，判定其語意欄位。\n"
    "規則：\n"
    "- intent 必須是提供的 enum 之一。\n"
    "- topic_tags 只能使用提供的 enum 值（繁體中文），不要發明新標籤。\n"
    "- stance_bias：使用者語氣看多=bullish、看空=bearish、中性=neutral。\n"
    "- 回答一律使用繁體中文標籤。"
)


class OpenAIStructuredQueryClassifier(QueryClassifier):
    """Responses API + structured outputs 分類器。失敗回傳 None，讓 InputLayer 降級。"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str = "",
        timeout_seconds: float = 8.0,
    ) -> None:
        self._api_key = api_key.strip()
        self._model_name = model_name
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._timeout = timeout_seconds
        self._schema = _build_schema()

    def classify(self, query_text: str) -> dict | None:
        payload = {
            "model": self._model_name,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": query_text}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ClassifyQuery",
                    "schema": self._schema,
                    "strict": True,
                }
            },
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                response_json = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        text = self._extract_text(response_json)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_text(payload: dict) -> str:
        collected: list[str] = []

        def walk(node) -> None:
            if isinstance(node, dict):
                if isinstance(node.get("text"), str):
                    collected.append(node["text"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return "\n".join(part for part in collected if part.strip())
