import json

import httpx

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent
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
from llm_stock_system.core.forecast import (
    apply_forecast_guardrail,
    apply_forecast_guardrail_to_list,
    build_forecast_block,
)
from llm_stock_system.core.target_price import (
    build_forward_price_fact,
    build_forward_price_highlight,
    build_forward_price_summary,
    is_forward_price_question,
)


class OpenAIResponsesSynthesisClient(LLMClient):
    """LLM-backed synthesizer.

    Phase 4: user prompt routes on ``intent`` + ``topic_tags``; ``question_type``
    is no longer included in any prompt sent to the LLM.
    """

    _INTENT_INSTRUCTIONS: dict[Intent, list[str]] = {
        Intent.NEWS_DIGEST: [
            "Focus: summarize the market sentiment and key event triggers from the evidence.",
            "If topic tags include '航運', emphasize freight-rate signals and analyst target-price reactions.",
            "If topic tags include '電價', emphasize electricity cost pressure and company response measures.",
            "If topic tags include '總經', emphasize macro indicators (CPI, yield) and institutional views.",
            "If topic tags include '法說', summarize positive and negative interpretations of the guidance.",
            "If topic tags include '上市', connect revenue data with post-listing price movement.",
        ],
        Intent.EARNINGS_REVIEW: [
            "Focus: summarize earnings quality, trend, and momentum from the financial evidence.",
            "If topic tags include '月營收', highlight MoM and YoY revenue change and whether it hit a recent high.",
            "If topic tags include '毛利率', explicitly address whether gross margin has turned positive and whether operating income followed.",
        ],
        Intent.VALUATION_CHECK: [
            "Focus: assess current valuation relative to historical range and peers.",
            "State the current PE ratio and its historical percentile position when evidence is available.",
            "If topic tags include '基本面', combine valuation with fundamental quality indicators.",
            "For forward price questions, anchor to analyst target prices or price-level evidence and do not fabricate numbers.",
        ],
        Intent.DIVIDEND_ANALYSIS: [
            "Focus: assess dividend yield, payout sustainability, and coverage ratios.",
            "If topic tags include '現金流', address whether free cash flow covers dividend payments over the past few years.",
            "If topic tags include '負債', address debt ratio trend and cash balance coverage of dividend obligations.",
        ],
        Intent.FINANCIAL_HEALTH: [
            "Focus: assess profitability stability, margin structure, and revenue growth.",
            "If a comparison ticker is present, compare the companies side by side using the most recent comparable period.",
            "Mention any loss years in the past 5 years and their likely cause if the evidence supports it.",
        ],
        Intent.TECHNICAL_VIEW: [
            "Focus: summarize price position relative to moving averages and key technical indicators.",
            "If topic tags include '籌碼', include margin balance and utilization rate alongside price data.",
            "Report RSI14, KD, MACD trend, and Bollinger position if available in the evidence.",
        ],
        Intent.INVESTMENT_ASSESSMENT: [
            "Focus: provide a balanced investment thesis combining fundamentals, valuation, and risk factors.",
            "Include at least three distinct risk reminders.",
            "If evidence covers both fundamental and valuation data, integrate both in the summary.",
        ],
    }
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
                    if query.is_forecast_query and preliminary_draft.forecast is None:
                        preliminary_draft.forecast = build_forecast_block(query, governance_report)
                    return preliminary_draft
                fallback = self._build_local_preliminary_fallback(query)
                if query.is_forecast_query:
                    fallback.forecast = build_forecast_block(query, governance_report)
                return fallback
            draft = self._fallback_client.synthesize(query, governance_report, system_prompt)
            if query.is_forecast_query:
                draft.forecast = build_forecast_block(query, governance_report)
            return draft

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

        user_prompt = self._build_grounded_user_prompt(query, evidence_lines)

        try:
            response_json = self._request(self._build_payload(system_prompt, user_prompt))
            text_output = self._extract_text(response_json)
            parsed = self._parse_json_block(text_output)
            parsed = self._apply_target_price_guardrails(query, governance_report, parsed)
            parsed = self._apply_fundamental_valuation_guardrails(query, governance_report, parsed)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            draft = self._fallback_client.synthesize(query, governance_report, system_prompt)
            if query.is_forecast_query:
                draft.forecast = build_forecast_block(query, governance_report)
            return draft

        summary = self._coerce_string(parsed.get("summary"), "資料不足，無法確認。")

        # Build forecast block for forecast queries and apply guardrails
        forecast_block = None
        highlights = self._coerce_list(parsed.get("highlights"))
        facts = self._coerce_list(parsed.get("facts"))
        if query.is_forecast_query:
            forecast_block = build_forecast_block(query, governance_report)
            summary = apply_forecast_guardrail(summary, forecast_block)
            highlights = apply_forecast_guardrail_to_list(highlights, forecast_block)
            facts = apply_forecast_guardrail_to_list(facts, forecast_block)

        return AnswerDraft(
            summary=summary,
            highlights=highlights,
            facts=facts,
            impacts=self._coerce_list(parsed.get("impacts")),
            risks=self._coerce_list(parsed.get("risks")),
            sources=sources,
            forecast=forecast_block,
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
        user_prompt = self._build_preliminary_user_prompt(query)

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

    # ------------------------------------------------------------------ #
    # Prompt builders                                                      #
    # ------------------------------------------------------------------ #

    def _build_grounded_user_prompt(self, query: StructuredQuery, evidence_lines: list[str]) -> str:
        return "\n".join(
            [
                *self._build_query_context_lines(query),
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
                "- highlights, facts, impacts, risks must be arrays of short strings.",
                "- Do not include markdown fences.",
            ]
        )

    def _build_preliminary_user_prompt(self, query: StructuredQuery) -> str:
        return "\n".join(
            [
                *self._build_query_context_lines(query),
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

    def _build_query_context_lines(self, query: StructuredQuery) -> list[str]:
        topic_tags = ", ".join(query.topic_tags) if query.topic_tags else "none"
        lines = [
            f"User query: {query.user_query}",
            f"Ticker: {query.ticker or 'unknown'}",
            f"Company: {query.company_name or 'unknown'}",
            f"Comparison ticker: {query.comparison_ticker or 'none'}",
            f"Comparison company: {query.comparison_company_name or 'none'}",
            f"Intent: {query.intent.value}",
            f"Topic tags: {topic_tags}",
        ]
        lines.extend(self._build_intent_instructions(query))
        return lines

    def _build_intent_instructions(self, query: StructuredQuery) -> list[str]:
        base = self._INTENT_INSTRUCTIONS.get(query.intent, [])
        tag_hint = f"Active topic tags: {', '.join(query.topic_tags)}" if query.topic_tags else ""
        lines = [line for line in ([tag_hint] + base) if line]

        # Inject forecast-specific hard rules
        if query.is_forecast_query:
            forecast_rules = [
                "FORECAST RULES (mandatory):",
                "- This is a forward-looking forecast query. Start the summary by stating the forecast window and date range.",
                "- Clearly distinguish between 'future scenario estimate' and 'historical fact'.",
                "- Do NOT present historical high/low points as future predictions.",
                "- Use scenario language: '情境推估', '可能', '偏向' — never '一定會', '肯定', '必定'.",
                "- State the direction tendency (偏多/偏空/區間震盪/無法判定) explicitly.",
                "- If using historical price data as proxy, explicitly say '以近期歷史波動作為代理參考'.",
                "- Do NOT fabricate specific price numbers or ranges. Only use numbers that appear in the evidence.",
                "- If no price range can be derived from evidence, say '目前無法提供具體價格區間' instead of inventing numbers.",
            ]
            if query.forecast_horizon_label:
                forecast_rules.append(f"- Forecast window label: {query.forecast_horizon_label}")
            lines.extend(forecast_rules)

        return lines

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
        if query.intent != Intent.VALUATION_CHECK or not is_forward_price_question(query):
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
