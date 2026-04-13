import json

import httpx

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.fundamental_valuation import (
    build_fundamental_valuation_facts,
    build_fundamental_valuation_highlights,
    build_fundamental_valuation_summary,
    is_fundamental_valuation_question,
)
from llm_stock_system.core.interfaces import LLMClient
from llm_stock_system.core.models import (
    AnswerDraft,
    GovernanceReport,
    SourceCitation,
    StructuredQuery,
)
from llm_stock_system.core.target_price import (
    build_forward_price_fact,
    build_forward_price_highlight,
    build_forward_price_summary,
    is_forward_price_question,
)


class OpenAIResponsesSynthesisClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "",
        preliminary_answers_enabled: bool = True,
    ) -> None:
        self._api_key = api_key.strip()
        self._model_name = model_name
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._preliminary_answers_enabled = preliminary_answers_enabled
        self._fallback_client = RuleBasedSynthesisClient()

    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
    ) -> AnswerDraft:
        if not governance_report.evidence:
            if self._preliminary_answers_enabled:
                preliminary_draft = self._synthesize_preliminary(query, system_prompt)
                if preliminary_draft is not None:
                    return preliminary_draft
                return self._build_local_preliminary_fallback(query)
            return self._fallback_client.synthesize(query, governance_report, system_prompt)

        sources = [
            SourceCitation(
                title=item.title,
                source_name=item.source_name,
                source_tier=item.source_tier,
                url=item.url,
                published_at=item.published_at,
                excerpt=item.excerpt,
                support_score=item.support_score,
                corroboration_count=item.corroboration_count,
            )
            for item in governance_report.evidence
        ]

        evidence_lines = [
            (
                f"- title={item.title}; source={item.source_name}; tier={item.source_tier.value}; "
                f"date={item.published_at:%Y-%m-%d}; excerpt={item.excerpt}"
            )
            for item in governance_report.evidence
        ]

        user_prompt = "\n".join(
            [
                f"User query: {query.user_query}",
                f"Ticker: {query.ticker or 'unknown'}",
                f"Company: {query.company_name or 'unknown'}",
                f"Comparison ticker: {query.comparison_ticker or 'none'}",
                f"Comparison company: {query.comparison_company_name or 'none'}",
                f"Topic: {query.topic.value}",
                f"Question type: {query.question_type}",
                "Evidence:",
                "\n".join(evidence_lines),
                "",
                "Return exactly one JSON object with these keys:",
                "summary, highlights, facts, impacts, risks",
                "Rules:",
                "- Use only the evidence provided above.",
                "- If evidence is insufficient, say: 資料不足，無法確認。",
                "- Respond in Traditional Chinese.",
                "- Use Traditional Chinese characters only; avoid Simplified Chinese wording.",
                "- Use Traditional Chinese characters only; avoid Simplified Chinese wording.",
                "- highlights, facts, impacts, risks must be arrays of short strings.",
                "- Do not include markdown fences.",
            ]
        )

        try:
            response_json = self._request(self._build_payload(system_prompt, user_prompt))
            text_output = self._extract_text(response_json)
            parsed = self._parse_json_block(text_output)
            parsed = self._apply_target_price_guardrails(query, governance_report, parsed)
            parsed = self._apply_fundamental_valuation_guardrails(query, governance_report, parsed)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return self._fallback_client.synthesize(query, governance_report, system_prompt)

        return AnswerDraft(
            summary=self._coerce_string(parsed.get("summary"), "資料不足，無法確認。"),
            highlights=self._coerce_list(parsed.get("highlights")),
            facts=self._coerce_list(parsed.get("facts")),
            impacts=self._coerce_list(parsed.get("impacts")),
            risks=self._coerce_list(parsed.get("risks")),
            sources=sources,
        )

    def _synthesize_preliminary(
        self,
        query: StructuredQuery,
        system_prompt: str,
    ) -> AnswerDraft | None:
        preliminary_system_prompt = "\n".join(
            [
                system_prompt,
                "",
                "Override for preliminary mode:",
                "- No grounded local evidence is currently available.",
                "- You may provide a tentative preliminary answer using general financial knowledge.",
                "- The summary must start with '初步判讀：'.",
                "- Clearly state that the answer is not based on retrieved local evidence.",
                "- Do not fabricate precise current figures, exact latest data, or nonexistent sources.",
                "- If the user asks about a fund or ETF using company-style financial metrics, explain the mismatch clearly.",
                "- Use cautious wording such as '可能', '通常', '傾向', '需再驗證'.",
                "- Respond in Traditional Chinese.",
            ]
        )
        user_prompt = "\n".join(
            [
                f"User query: {query.user_query}",
                f"Ticker: {query.ticker or 'unknown'}",
                f"Company: {query.company_name or 'unknown'}",
                f"Comparison ticker: {query.comparison_ticker or 'none'}",
                f"Comparison company: {query.comparison_company_name or 'none'}",
                f"Topic: {query.topic.value}",
                f"Question type: {query.question_type}",
                "",
                "Return exactly one JSON object with these keys:",
                "summary, highlights, facts, impacts, risks",
                "Rules:",
                "- This is a preliminary answer because no local evidence was retrieved.",
                "- Do not claim that the answer is verified.",
                "- Avoid exact figures unless they are essential and you are highly confident.",
                "- highlights, facts, impacts, risks must be arrays of short strings.",
                "- Do not include markdown fences.",
            ]
        )

        try:
            response_json = self._request(self._build_payload(preliminary_system_prompt, user_prompt))
            text_output = self._extract_text(response_json)
            parsed = self._parse_json_block(text_output)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return None

        summary = self._coerce_string(parsed.get("summary"), "")
        if summary and not summary.startswith("初步判讀："):
            summary = f"初步判讀：{summary}"

        return AnswerDraft(
            summary=summary or "初步判讀：目前本地資料庫尚未補齊，以下僅能提供低信心的初步方向判斷。",
            highlights=self._coerce_list(parsed.get("highlights"))
            or ["以下內容為 LLM 在本地缺乏證據時提供的初步判讀，尚待資料同步後驗證。"],
            facts=self._coerce_list(parsed.get("facts"))
            or ["目前本地資料庫尚未取得足夠證據，因此以下內容不屬於已驗證的 grounded 結論。"],
            impacts=self._coerce_list(parsed.get("impacts")),
            risks=self._coerce_list(parsed.get("risks"))
            or [
                "這段回覆不是根據本地已驗證資料生成，應視為低信心初步判讀。",
                "正式判斷仍應等待資料同步完成後，再回到 grounded 結果確認。",
            ],
            sources=[],
        )

    def _build_local_preliminary_fallback(self, query: StructuredQuery) -> AnswerDraft:
        label = query.company_name or query.ticker or "此標的"
        if query.comparison_company_name or query.comparison_ticker:
            comparison_label = query.comparison_company_name or query.comparison_ticker
            label = f"{label}、{comparison_label}"

        return AnswerDraft(
            summary=(
                f"初步判讀：{label} 目前在本地資料庫尚未取得足夠可驗證證據，"
                "系統已先改用低信心模式提供方向性整理，後續仍需等待新聞與公告資料同步後再確認。"
            ),
            highlights=[
                "目前查詢已進入初步判讀模式，代表本地尚未取得足夠 grounded 證據。",
                "若外部資料源稍後同步完成，再次查詢時通常會優先回到本地證據式回答。",
            ],
            facts=[
                f"查詢主體為 {label}，但目前本地資料庫未能形成足夠證據鏈。",
                "OpenAI preliminary 已嘗試啟用；若外部模型或資料源當下不可用，系統會改回本地低信心初步整理。",
            ],
            impacts=[
                "這能避免使用者只看到完全空白的資料不足訊息。",
                "但因缺少本地證據，這段內容不應視為正式結論。",
            ],
            risks=[
                "這段回覆不是根據本地已驗證資料生成，應視為低信心初步判讀。",
                "正式判斷仍應等待資料同步完成後，再回到 grounded 結果確認。",
                "若查詢涉及即時新聞或市場情緒，缺少原始來源時更容易出現偏差。",
            ],
            sources=[],
        )

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self._model_name,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        }

    def _request(self, payload: dict) -> dict:
        with httpx.Client(timeout=25.0) as client:
            response = client.post(
                f"{self._base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _extract_text(self, payload: dict) -> str:
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

    def _parse_json_block(self, text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(stripped[start : end + 1])
        raise ValueError("OpenAI response did not contain a JSON object")

    def _apply_target_price_guardrails(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        parsed: dict,
    ) -> dict:
        if query.question_type != "price_outlook" or not is_forward_price_question(query):
            return parsed

        guarded = dict(parsed)
        guarded["summary"] = build_forward_price_summary(query, governance_report)

        highlights = self._coerce_list(parsed.get("highlights"))
        facts = self._coerce_list(parsed.get("facts"))

        target_highlight = build_forward_price_highlight(query, governance_report)
        target_fact = build_forward_price_fact(query, governance_report)

        guarded["highlights"] = [target_highlight, *highlights][:3]
        guarded["facts"] = [target_fact, *facts][:3]
        return guarded

    def _apply_fundamental_valuation_guardrails(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        parsed: dict,
    ) -> dict:
        if not is_fundamental_valuation_question(query):
            return parsed

        guarded = dict(parsed)
        guarded["summary"] = build_fundamental_valuation_summary(query, governance_report)
        guarded["highlights"] = build_fundamental_valuation_highlights(query, governance_report)
        guarded["facts"] = build_fundamental_valuation_facts(query, governance_report)
        return guarded

    def _coerce_list(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _coerce_string(self, value, default: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default
