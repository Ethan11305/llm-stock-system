from __future__ import annotations

import json

import httpx

from llm_stock_system.core.enums import Intent, StanceBias, TopicTag
from llm_stock_system.core.interfaces import QueryClassifier
from llm_stock_system.core.models import QUESTION_TYPE_TO_INTENT


_TIME_RANGE_LABELS = ["1d", "7d", "30d", "latest_quarter", "1y", "3y", "5y"]


def _build_schema() -> dict:
    """Derive the classification schema from existing source-of-truth dicts.

    Keeps the enum lists in sync with QUESTION_TYPE_TO_INTENT / TopicTag / Intent
    automatically — no hand-maintained duplicate.
    """
    question_types = sorted(QUESTION_TYPE_TO_INTENT.keys())
    intents = [i.value for i in Intent]
    topic_tag_values = [t.value for t in TopicTag]
    stance_values = [s.value for s in StanceBias]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent",
            "question_type",
            "topic_tags",
            "time_range_label",
            "stance_bias",
            "is_forecast_query",
            "wants_direction",
            "wants_scenario_range",
            "forecast_horizon_label",
            "forecast_horizon_days",
        ],
        "properties": {
            "intent": {"type": "string", "enum": intents},
            "question_type": {"type": "string", "enum": question_types},
            "topic_tags": {
                "type": "array",
                "items": {"type": "string", "enum": topic_tag_values},
            },
            "time_range_label": {"type": "string", "enum": _TIME_RANGE_LABELS},
            "stance_bias": {"type": "string", "enum": stance_values},
            "is_forecast_query": {"type": "boolean"},
            "wants_direction": {"type": "boolean"},
            "wants_scenario_range": {"type": "boolean"},
            "forecast_horizon_label": {"type": ["string", "null"]},
            "forecast_horizon_days": {"type": ["integer", "null"]},
        },
    }


_SYSTEM_PROMPT = (
    "你是台股查詢分類器。根據使用者的自由文字問題，判定其語意欄位。\n"
    "規則：\n"
    "- intent 與 question_type 必須對應（例如 price_outlook → valuation_check）。\n"
    "- topic_tags 只能使用提供的 enum 值（繁體中文），不要發明新標籤。\n"
    "- 若問題提到「未來/下週/下半年」等前瞻性時間窗，且帶有方向或區間需求，"
    "is_forecast_query 應為 true，並設定對應的 forecast_horizon_label/days。\n"
    "- 若僅詢問歷史或當前狀態，is_forecast_query 為 false。\n"
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
