import re

from llm_stock_system.core.enums import Intent, SourceTier, StanceBias, Topic, TopicTag
from llm_stock_system.core.fundamental_valuation import (
    build_fundamental_valuation_facts,
    build_fundamental_valuation_highlights,
    build_fundamental_valuation_summary,
    is_fundamental_valuation_question,
)
from llm_stock_system.core.interfaces import LLMClient
from llm_stock_system.core.models import AnswerDraft, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.target_price import (
    build_forward_price_fact,
    build_forward_price_highlight,
    build_forward_price_summary,
    is_forward_price_question,
)


class RuleBasedSynthesisClient(LLMClient):
    """Fallback synthesizer that keeps answers grounded in retrieved evidence.

    Phase 4 routing:
    - Primary router: ``query.intent``  (7 Intent values)
    - Secondary router: ``query.topic_tags`` (matched tags, free keywords, and fallback tags merged)
    - ``question_type`` is no longer read anywhere in this class.
    """

    # ------------------------------------------------------------------ #
    # Intent dispatch tables                                               #
    # ------------------------------------------------------------------ #
    _SUMMARY_BUILDERS: dict[Intent, str] = {
        Intent.NEWS_DIGEST:           "_build_summary_news_digest",
        Intent.EARNINGS_REVIEW:       "_build_summary_earnings_review",
        Intent.VALUATION_CHECK:       "_build_summary_valuation_check",
        Intent.DIVIDEND_ANALYSIS:     "_build_summary_dividend_analysis",
        Intent.FINANCIAL_HEALTH:      "_build_summary_financial_health",
        Intent.TECHNICAL_VIEW:        "_build_summary_technical_view",
        Intent.INVESTMENT_ASSESSMENT: "_build_summary_investment_assessment",
    }
    _IMPACTS_BUILDERS: dict[Intent, str] = {
        Intent.NEWS_DIGEST:           "_build_impacts_news_digest",
        Intent.EARNINGS_REVIEW:       "_build_impacts_earnings_review",
        Intent.VALUATION_CHECK:       "_build_impacts_valuation_check",
        Intent.DIVIDEND_ANALYSIS:     "_build_impacts_dividend_analysis",
        Intent.FINANCIAL_HEALTH:      "_build_impacts_financial_health",
        Intent.TECHNICAL_VIEW:        "_build_impacts_technical_view",
        Intent.INVESTMENT_ASSESSMENT: "_build_impacts_investment_assessment",
    }
    _RISKS_BUILDERS: dict[Intent, str] = {
        Intent.NEWS_DIGEST:           "_build_risks_news_digest",
        Intent.EARNINGS_REVIEW:       "_build_risks_earnings_review",
        Intent.VALUATION_CHECK:       "_build_risks_valuation_check",
        Intent.DIVIDEND_ANALYSIS:     "_build_risks_dividend_analysis",
        Intent.FINANCIAL_HEALTH:      "_build_risks_financial_health",
        Intent.TECHNICAL_VIEW:        "_build_risks_technical_view",
        Intent.INVESTMENT_ASSESSMENT: "_build_risks_investment_assessment",
    }

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
    ) -> AnswerDraft:
        _ = system_prompt

        if not governance_report.evidence:
            return AnswerDraft(
                summary="資料不足，無法確認。",
                highlights=["現有證據不足或一致性不足，系統已降級回答。"],
                facts=["尚未取得足夠的官方公告、新聞或財報資料。"],
                impacts=["資料不足時，不應將單一訊息直接解讀為趨勢。"],
                risks=[
                    "資料不足時，容易誤把單一訊息當成趨勢。",
                    "若只依賴未驗證資訊，可能放大判斷偏誤。",
                    "建議等待更多公告、財報或主流來源更新。",
                ],
                sources=[],
            )

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

        # 預設 highlights：優先顯示 excerpt 摘要，若無 excerpt 才退回標題
        highlights = [
            item.excerpt[:120] if item.excerpt and len(item.excerpt) > 20
            else f"{item.title}（{item.source_name}）"
            for item in governance_report.evidence[:3]
        ]
        facts = [
            f"{item.source_name} 於 {item.published_at:%Y-%m-%d} 提供資料：{item.excerpt}"
            for item in governance_report.evidence[:3]
        ]
        if is_fundamental_valuation_question(query):
            highlights = build_fundamental_valuation_highlights(query, governance_report)
            facts = build_fundamental_valuation_facts(query, governance_report)
        # earnings / EPS 相關類型也用 excerpt 作為 highlights
        elif query.question_type in {
            "earnings_summary", "eps_dividend_review", "revenue_growth_review",
            "monthly_revenue_yoy_review", "profitability_stability_review",
            "gross_margin_comparison_review", "season_line_margin_review",
        }:
            highlights = [
                item.excerpt[:120] if item.excerpt else f"{item.title}（{item.source_name}）"
                for item in governance_report.evidence[:3]
            ]
        if query.intent == Intent.VALUATION_CHECK and is_forward_price_question(query):
            highlights = [build_forward_price_highlight(query, governance_report), *highlights][:3]
            facts = [build_forward_price_fact(query, governance_report), *facts][:3]

        return AnswerDraft(
            summary=self._build_summary(query, governance_report),
            highlights=highlights,
            facts=facts,
            impacts=self._build_impacts(query),
            risks=self._build_risks(query, governance_report),
            sources=sources,
        )

    # ------------------------------------------------------------------ #
    # Top-level dispatch                                                   #
    # ------------------------------------------------------------------ #

    def _build_summary(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        builder_name = self._SUMMARY_BUILDERS.get(query.intent)
        if builder_name:
            return getattr(self, builder_name)(query, governance_report)
        return self._build_summary_fallback(query, governance_report)

    def _build_impacts(self, query: StructuredQuery) -> list[str]:
        builder_name = self._IMPACTS_BUILDERS.get(query.intent)
        if builder_name:
            return getattr(self, builder_name)(query)
        return self._build_impacts_generic()

    def _build_risks(self, query: StructuredQuery, governance_report: GovernanceReport) -> list[str]:
        builder_name = self._RISKS_BUILDERS.get(query.intent)
        if builder_name:
            risks = getattr(self, builder_name)(query)
        else:
            risks = self._build_risks_generic(query)

        if governance_report.high_trust_ratio < 0.5:
            risks.append("目前高可信來源占比不高，建議回看原文再做判斷。")

        return risks[:4]

    # ================================================================== #
    # SUMMARY builders — one per Intent                                   #
    # ================================================================== #

    _THEME_TAGS: frozenset[str] = frozenset({
        "題材", "產業",
        TopicTag.EV.value,
        TopicTag.AI.value,
        TopicTag.SEMICON_EQUIP.value,
    })

    def _build_summary_news_digest(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.SHIPPING.value in tags or "SCFI" in tags:
            return self._summarize_shipping(label, query, governance_report)
        if TopicTag.ELECTRICITY.value in tags:
            return self._summarize_electricity(label, query, governance_report)
        if TopicTag.MACRO.value in tags or "殖利率" in tags or "CPI" in tags:
            return self._summarize_macro_yield(label, governance_report)
        # theme 必須在 guidance 之前：半導體設備/AI/電動車 題材查詢可能同時帶有「法說」tag
        if tags & self._THEME_TAGS:
            return self._summarize_theme_impact(label, query, governance_report)
        if TopicTag.GUIDANCE.value in tags or "指引" in tags:
            return self._summarize_guidance_reaction(label, governance_report)
        if TopicTag.LISTING.value in tags:
            return self._summarize_listing_revenue(label, query, governance_report)

        return self._build_summary_fallback(query, governance_report)

    def _build_summary_earnings_review(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.GROSS_MARGIN.value in tags or "轉正" in tags:
            return self._summarize_margin_turnaround(label, governance_report)
        if TopicTag.REVENUE.value in tags or "月增率" in tags or "年增率" in tags:
            return self._summarize_monthly_revenue_yoy(label, governance_report)
        if "EPS" in tags or "財報" in tags:
            return self._summarize_earnings_eps(label, governance_report)

        # 通用財報摘要
        annual_eps = self._extract_number(governance_report, r"全年 EPS 約 (\d+(?:\.\d+)?) 元")
        latest_quarter_eps = self._extract_number(governance_report, r"最新一季 EPS 約 (\d+(?:\.\d+)?) 元")
        if annual_eps:
            return f"{label} 目前可整理到的財報重點顯示，去年全年 EPS 約 {annual_eps} 元。"
        if latest_quarter_eps:
            return f"{label} 目前可整理到的最新一季財報重點顯示， EPS 約 {latest_quarter_eps} 元。"
        return f"{label} 目前有財報相關資料可供整理，但細節仍應回看原始揭露內容。"

    def _build_summary_valuation_check(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if is_fundamental_valuation_question(query):
            return build_fundamental_valuation_summary(query, governance_report)
        if is_forward_price_question(query):
            return build_forward_price_summary(query, governance_report)
        if "股價區間" in tags:
            return self._summarize_price_range(label, query, governance_report)
        if TopicTag.VALUATION.value in tags or "本益比" in tags:
            return self._summarize_pe_valuation(label, governance_report)

        return self._build_summary_fallback(query, governance_report)

    def _build_summary_dividend_analysis(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if "除息" in tags or "填息" in tags:
            return self._summarize_ex_dividend_performance(label, governance_report)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return self._summarize_fcf_dividend_sustainability(label, governance_report)
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return self._summarize_debt_dividend_safety(label, governance_report)

        # 通用：股利殖利率
        return self._summarize_dividend_yield(label, governance_report)

    def _build_summary_financial_health(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if (TopicTag.GROSS_MARGIN.value in tags or "毛利率" in tags) and (
            query.comparison_ticker or query.comparison_company_name
        ):
            return self._summarize_gross_margin_comparison(label, query, governance_report)
        if "獲利" in tags or "穩定性" in tags:
            return self._summarize_profitability_stability(label, governance_report)
        if "成長" in tags or "AI" in tags or "伺服器" in tags:
            return self._summarize_revenue_growth(label, governance_report)

        return self._build_summary_fallback(query, governance_report)

    def _build_summary_technical_view(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return self._summarize_season_line_margin(label, governance_report)

        # 通用：技術指標
        return self._summarize_technical_indicators(label, governance_report)

    def _build_summary_investment_assessment(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if "公告" in tags:
            lead_title = governance_report.evidence[0].title if governance_report.evidence else "最新公告"
            return f"{label} 近期有可追蹤的公告或高可信資料更新，重點包括：{lead_title}。"

        if is_fundamental_valuation_question(query):
            return build_fundamental_valuation_summary(query, governance_report)

        return self._build_summary_fallback(query, governance_report)

    def _build_summary_fallback(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        label = query.company_name or query.ticker or "此標的"
        official_count = sum(1 for item in governance_report.evidence if item.source_tier == SourceTier.HIGH)
        if official_count:
            return f"{label} 目前有官方或高可信來源可供整理，但現有資訊仍應與後續公告一併觀察。"
        return f"{label} 目前有部分可參考資料，但來源強度與一致性仍需持續確認。"

    # ================================================================== #
    # SUMMARY sub-routines (content identical to old question_type blocks)#
    # ================================================================== #

    def _summarize_margin_turnaround(self, label: str, governance_report: GovernanceReport) -> str:
        latest_margin = self._extract_number(governance_report, r"毛利率約 (-?\d+(?:\.\d+)?)%")
        previous_margin = self._extract_number(governance_report, r"上一季（\d{4}-\d{2}-\d{2}）毛利率約 (-?\d+(?:\.\d+)?)%")
        latest_operating_income = self._extract_number(governance_report, r"最新季營業利益約 (-?\d+(?:\.\d+)?) 億元")
        previous_operating_income = self._extract_number(governance_report, r"上一季營業利益約 (-?\d+(?:\.\d+)?) 億元")
        gross_margin_status = self._extract_text(
            governance_report,
            r"(毛利率已由負轉正|毛利率未出現由負轉正，仍維持正值|毛利率仍為負值，尚未轉正|毛利率由正轉負)",
        )
        operating_status = self._extract_text(
            governance_report,
            r"(營業利益已同步轉正|營業利益維持正值|營業利益由正轉負|營業利益尚未同步轉正)",
        )
        profitability_view = self._extract_text(
            governance_report,
            r"(最新季毛利率已由負轉正，且營業利益也同步轉正，目前可視為本業層面的實質獲利改善|最新季毛利率雖已由負轉正，但營業利益仍未同步轉正，較像本業修復初期，暫時還不能直接視為實質獲利改善|最新季毛利率仍維持正值，但營業利益依舊為負，本業獲利修復仍不完整，目前尚難視為實質獲利改善|雖然營業利益已轉正，但毛利率仍未回到正值，需留意是否有一次性因素扭曲本業判讀|毛利率與營業利益都尚未同時站穩正值，目前仍難視為實質獲利改善)",
        )
        if latest_margin is not None and latest_operating_income is not None and gross_margin_status and operating_status:
            gm_text = gross_margin_status if gross_margin_status.endswith("。") else f"{gross_margin_status}。"
            op_text = operating_status if operating_status.endswith("。") else f"{operating_status}。"
            pv_text = profitability_view or ""
            if pv_text and not pv_text.endswith("。"):
                pv_text = f"{pv_text}。"
            summary = f"{label} 最新季毛利率約 {latest_margin}%，{gm_text}營業利益約 {latest_operating_income} 億元，{op_text}"
            if previous_margin is not None and previous_operating_income is not None:
                summary += f"上一季毛利率約 {previous_margin}%，營業利益約 {previous_operating_income} 億元。"
            if pv_text:
                summary += pv_text
            return summary
        return f"資料不足，無法確認{label}最新季毛利率是否由負轉正，以及營業利益是否同步轉正。"

    def _summarize_listing_revenue(self, label: str, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        listing_event_label = self._listing_event_label(query)
        revenue_month = self._extract_text(governance_report, r"最新已公布月營收為 (\d{4}-\d{2})")
        month_revenue = self._extract_number(governance_report, r"單月營收約 (\d+(?:\.\d+)?) 億元")
        mom_pct = self._extract_number(governance_report, r"月增率約 (-?\d+(?:\.\d+)?)%")
        yoy_pct = self._extract_number(governance_report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        has_route_news = self._evidence_contains(governance_report, ("航線", "熊本", "旅展", "機票"))
        has_fee_news = self._evidence_contains(governance_report, ("燃油附加費", "票價", "附加費"))
        has_brand_news = self._evidence_contains(governance_report, ("A350", "空山基", "機上餐飲", "品牌"))
        if revenue_month and month_revenue:
            parts = [f"{label} {listing_event_label}的股價波動，現有資料較像市場對營運題材與消息面交互反應。"]
            parts.append(f"最新已公布月營收為 {revenue_month}，單月營收約 {month_revenue} 億元。")
            if mom_pct is not None:
                parts.append(f"月增率約 {mom_pct}%。")
            if yoy_pct is not None:
                parts.append(f"年增率約 {yoy_pct}%。")
            if yoy_pct is not None and float(yoy_pct) >= 30:
                parts.append("若以年增率觀察，這屬偏強的營收成長訊號。")
            elif mom_pct is not None and float(mom_pct) >= 20:
                parts.append("若以月增率觀察，這屬短線偏強的營收增長訊號。")
            else:
                parts.append("但現有數據仍不足以確認有單一重大營收利多可完整解釋股價波動。")
            if has_route_news or has_fee_news or has_brand_news:
                focus_parts: list[str] = []
                if has_route_news:
                    focus_parts.append("新航線與促銷活動")
                if has_fee_news:
                    focus_parts.append("燃油附加費或票價調整")
                if has_brand_news:
                    focus_parts.append("品牌與服務話題")
                parts.append(f"近期新聞焦點較多落在{'、'.join(focus_parts)}。")
            return "".join(parts)
        return f"資料不足，無法確認{label}{listing_event_label}股價波動的主因，以及是否有重大營收增長消息。"

    def _summarize_guidance_reaction(self, label: str, governance_report: GovernanceReport) -> str:
        positive_count = self._extract_number(governance_report, r"正面解讀共有 (\d+) 則")
        negative_count = self._extract_number(governance_report, r"負面解讀共有 (\d+) 則")
        has_positive = positive_count is not None or self._evidence_contains(
            governance_report, ("正面反應整理", "看好", "調高", "上修")
        )
        has_negative = negative_count is not None or self._evidence_contains(
            governance_report, ("負面反應整理", "保守", "下修", "不如預期")
        )
        if has_positive and has_negative:
            return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正負並存；目前可整理到同時存在正面與負面反應，顯示市場對後續動能仍有分歧。"
        if has_positive:
            return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正面；現有中高可信來源較多聚焦於成長動能或展望支撐。"
        if has_negative:
            return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向保守；現有來源較多聚焦於下修、壓力或不確定性。"
        return f"資料不足，無法確認{label}法說會後媒體與法人對下半年營運指引的正負面反應。"

    def _summarize_monthly_revenue_yoy(self, label: str, governance_report: GovernanceReport) -> str:
        availability_sentence = self._extract_text(
            governance_report,
            r"(截至 \d{4}-\d{2}-\d{2}，官方月營收資料最新僅到 \d{4}-\d{2}，尚未公布 \d{4}-\d{2} 月營收。[^。]*。)",
        )
        if availability_sentence:
            return availability_sentence

        month_revenue = self._extract_number(governance_report, r"單月營收約 (\d+(?:\.\d+)?) 億元")
        mom_pct = self._extract_number(governance_report, r"月增率約 (-?\d+(?:\.\d+)?)%")
        yoy_pct = self._extract_number(governance_report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        mom_status = self._extract_text(governance_report, r"(月增率已超過 20%|月增率未達 20%)")
        high_status = self._extract_text(
            governance_report,
            r"(創下近一年新高|尚未創下近一年新高|近一年月營收歷史不足[^。]*。)",
        )
        market_view = self._extract_text(governance_report, r"(市場解讀：[^。]*。)")

        if month_revenue is not None and mom_pct is not None:
            parts = [f"{label} 最新已取得的單月營收約 {month_revenue} 億元，月增率約 {mom_pct}%。"]
            if mom_status:
                parts.append(mom_status + "。")
            if yoy_pct is not None:
                parts.append(f"年增率約 {yoy_pct}%。")
            if high_status:
                parts.append(high_status if high_status.endswith("。") else high_status + "。")
            if market_view:
                parts.append(market_view)
            return "".join(parts)

        cumulative_month_count = self._extract_number(governance_report, r"\d{4} 年前 (\d+) 個月累計營收約")
        current_total = self._extract_number(governance_report, r"\d{4} 年前 \d+ 個月累計營收約 (\d+(?:\.\d+)?) 億元")
        previous_total = self._extract_number(governance_report, r"\d{4} 年同期約 (\d+(?:\.\d+)?) 億元")
        cumulative_yoy_pct = self._extract_number(governance_report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        cumulative_availability = self._extract_text(
            governance_report,
            r"(截至 \d{4}-\d{2}，官方月營收資料僅更新到今年前 \d+ 個月。)",
        )
        if current_total is not None and previous_total is not None and cumulative_yoy_pct is not None:
            leading = cumulative_availability or ""
            month_label = int(cumulative_month_count) if cumulative_month_count is not None else 3
            return (
                f"{leading}{label} 今年前 {month_label} 個月累計營收約 {current_total} 億元，"
                f"去年同期約 {previous_total} 億元，年增率約 {cumulative_yoy_pct}%。"
            )
        return f"資料不足，無法確認{label}的月營收變化、單月月增率或與去年同期相比的成長幅度。"

    def _summarize_price_range(self, label: str, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        high_price, low_price = self._extract_price_range(governance_report)
        if high_price and low_price:
            return f"{label} 近 {query.time_range_days} 天資料顯示，最高價為 {high_price} 元，最低價為 {low_price} 元。"
        return f"{label} 目前已有股價資料，但不足以穩定整理出完整區間。"

    def _summarize_gross_margin_comparison(
        self, label: str, query: StructuredQuery, governance_report: GovernanceReport
    ) -> str:
        primary_label = query.company_name or query.ticker or "第一家公司"
        comparison_label = query.comparison_company_name or query.comparison_ticker or "第二家公司"
        primary_margin = self._extract_company_margin(governance_report, primary_label)
        comparison_margin = self._extract_company_margin(governance_report, comparison_label)
        higher_company = self._extract_text(
            governance_report,
            rf"由\s*({re.escape(primary_label)}|{re.escape(comparison_label)})\s*較高",
        )
        margin_gap = self._extract_number(governance_report, r"高出 (\d+(?:\.\d+)?) 個百分點")
        if primary_margin and comparison_margin:
            resolved_higher = higher_company or (
                primary_label if float(primary_margin) >= float(comparison_margin) else comparison_label
            )
            gap_segment = f"，高出 {margin_gap} 個百分點" if margin_gap is not None else ""
            return (
                f"若以最新可比財報口徑比較，"
                f"{primary_label} 毛利率約 {primary_margin}%，"
                f"{comparison_label} 毛利率約 {comparison_margin}%，"
                f"由 {resolved_higher} 較高{gap_segment}。"
                f"單看毛利率，{resolved_higher} 在定價能力或成本結構上相對較占優勢，"
                "但不能單憑毛利率就等同整體經營效率更好。"
            )
        return (
            f"資料不足，無法確認{primary_label}與{comparison_label}"
            "最新可比財報口徑下的毛利率差異與經營效率評估。"
        )

    def _summarize_profitability_stability(self, label: str, governance_report: GovernanceReport) -> str:
        stability_sentence = self._extract_text(
            governance_report,
            r"(近五年[^。]*(?:穩定獲利|波動[^。]*|並非每年都有穩定獲利)[^。]*。)",
        )
        loss_year = self._extract_number(governance_report, r"在 (\d{4}) 年歸屬母公司淨損約")
        loss_amount = self._extract_number(governance_report, r"淨損約 (-?\d+(?:\.\d+)?) 億元")
        loss_reason = self._extract_text(governance_report, r"(若只看財報結構推估[^。]*。)")
        if stability_sentence and loss_year and loss_amount:
            tail = f"{loss_reason}" if loss_reason else ""
            normalized = str(abs(float(loss_amount)))
            return f"{stability_sentence}{loss_year} 年是較明顯的虧損年度，歸屬母公司淨損約 {normalized} 億元。{tail}"
        if stability_sentence:
            return stability_sentence
        return f"資料不足，無法確認{label}過去五年的獲利穩定性與轉虧年度原因。"

    def _summarize_debt_dividend_safety(self, label: str, governance_report: GovernanceReport) -> str:
        debt_ratio = self._extract_number(governance_report, r"負債比率約 (\d+(?:\.\d+)?)%")
        previous_ratio = self._extract_number(governance_report, r"前一季約 (\d+(?:\.\d+)?)%")
        year_ago_ratio = self._extract_number(governance_report, r"去年同期約 (\d+(?:\.\d+)?)%")
        debt_status = self._extract_text(
            governance_report,
            r"負債比率(未見突然升高，反而較前幾期回落|未見突然升高，大致持平|近期沒有明顯異常升高|有溫和升高|有明顯升高)",
        )
        cash_balance = self._extract_number(governance_report, r"現金及約當現金約 (\d+(?:\.\d+)?) 億元")
        dividend_total = self._extract_number(governance_report, r"現金股利發放總額約 (\d+(?:\.\d+)?) 億元")
        coverage_ratio = self._extract_number(governance_report, r"約可覆蓋 (\d+(?:\.\d+)?) 倍")
        payout_view = self._extract_text(
            governance_report,
            r"(若只看帳上現金，現金部位看起來足以支應現金股利|若只看帳上現金，現金部位大致可支應現金股利，但仍要留意後續營運與資本支出|若只看帳上現金，現金部位勉強可支應現金股利，但緩衝不算特別厚|若只看帳上現金，支應現金股利的緩衝偏薄)",
        )
        if debt_ratio and debt_status and cash_balance and dividend_total and coverage_ratio:
            previous_segment = ""
            if previous_ratio and year_ago_ratio:
                previous_segment = f"；前一季約 {previous_ratio}%，去年同期約 {year_ago_ratio}%"
            payout_segment = (
                f"；最新現金及約當現金約 {cash_balance} 億元，"
                f"對最近一次現金股利總額約 {dividend_total} 億元，約可覆蓋 {coverage_ratio} 倍"
            )
            payout_tail = f"，{payout_view}" if payout_view else ""
            return (
                f"{label} 最新負債比率約 {debt_ratio}%{previous_segment}，{debt_status}"
                f"{payout_segment}{payout_tail}。"
            )
        return f"資料不足，無法確認{label}負債比率是否突然升高，以及現金部位是否足以支應股利。"

    def _summarize_fcf_dividend_sustainability(self, label: str, governance_report: GovernanceReport) -> str:
        annual_fcf = self._extract_year_metric_pairs(
            governance_report,
            r"(\d{4}) 年營業活動淨現金流入約 [\d.]+ 億元，資本支出約 [\d.]+ 億元，推估自由現金流約 ([\d.]+) 億元",
        )
        annual_payout = self._extract_year_metric_pairs(
            governance_report,
            r"(\d{4}) 年現金股利每股約 [\d.]+ 元。依參與分派總股數約 [\d.]+ 股估算，現金股利發放總額約 ([\d.]+) 億元",
        )
        sustainability = self._extract_text(
            governance_report,
            r"(近三年自由現金流均高於現金股利支出，顯示目前股利政策具一定永續性|近三年大致能以自由現金流支應現金股利，整體永續性偏穩健|自由現金流對股利支應能力接近打平，後續仍需留意資本支出與獲利變化|自由現金流對現金股利支應能力偏弱，股利政策永續性需保守看待)",
        )
        if annual_fcf and annual_payout:
            fcf_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_fcf[:3])
            payout_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_payout[:3])
            conclusion = sustainability or "目前仍需持續觀察資本支出與獲利變化"
            return (
                f"{label} 近三個已揭露股利年度的自由現金流約為 {fcf_segment}；"
                f"現金股利發放總額約為 {payout_segment}。"
                f"就目前公開資料看，{conclusion}。"
            )
        if annual_fcf:
            fcf_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_fcf[:3])
            return f"{label} 近三年自由現金流約為 {fcf_segment}，但現金股利發放總額資料仍不足以完整評估股利政策永續性。"
        return f"資料不足，無法確認{label}過去三年的自由現金流、現金股利發放總額與股利政策永續性。"

    def _summarize_pe_valuation(self, label: str, governance_report: GovernanceReport) -> str:
        current_pe = self._extract_number(governance_report, r"本益比約 (\d+(?:\.\d+)?) 倍")
        low_pe = self._extract_number(governance_report, r"本益比區間約 (\d+(?:\.\d+)?) 至")
        high_pe = self._extract_number(governance_report, r"至 (\d+(?:\.\d+)?) 倍")
        percentile = self._extract_number(governance_report, r"歷史分位 (\d+(?:\.\d+)?)%")
        valuation_zone = self._extract_text(governance_report, r"屬(歷史偏高區|歷史偏低區|歷史中段區)")
        entry_view = self._extract_text(governance_report, r"對長期投資來說，(.+?)。")
        if current_pe and low_pe and high_pe and valuation_zone:
            tail = f"；若以近 13 個月歷史區間衡量，{entry_view}" if entry_view else ""
            return f"{label} 目前本益比約 {current_pe} 倍；若看近 13 個月區間約 {low_pe} 至 {high_pe} 倍，目前屬{valuation_zone}{tail}。"
        if current_pe and percentile:
            return f"{label} 目前本益比約 {current_pe} 倍，約落在近 13 個月歷史分位 {percentile}% 左右，但仍需搭配獲利成長與產業狀況一起判斷是否偏貴。"
        return f"資料不足，無法確認{label}目前本益比在歷史區間所處的位置與是否偏貴。"

    def _summarize_revenue_growth(self, label: str, governance_report: GovernanceReport) -> str:
        share_value = self._extract_number(governance_report, r"(\d+(?:\.\d+)?)%")
        has_ai_server = self._evidence_contains(governance_report, ("ai伺服器", "ai 伺服器", "伺服器", "server"))
        has_growth_2026 = self._evidence_contains(governance_report, ("2026", "倍增", "成長", "動能"))
        if has_ai_server and has_growth_2026 and share_value:
            return f"{label} 目前公開資訊顯示，AI 伺服器相關營收比重約 {share_value}%，且 2026 年成長仍被多個來源視為重要動能。"
        if has_ai_server and has_growth_2026:
            return f"{label} 目前公開資訊顯示，AI 伺服器仍被視為 2026 年的重要成長動能，但現有來源未一致揭露明確營收占比。"
        if has_ai_server:
            return f"{label} 目前已有 AI 伺服器相關資訊可供整理，但對 2026 年成長預測與營收占比仍缺乏足夠一致的公開證據。"
        return f"資料不足，無法確認{label}AI 伺服器營收占比與 2026 年成長預測。"

    def _summarize_technical_indicators(self, label: str, governance_report: GovernanceReport) -> str:
        rsi = self._extract_number(governance_report, r"RSI14 約 (\d+(?:\.\d+)?)")
        k_value = self._extract_number(governance_report, r"K 值約 (\d+(?:\.\d+)?)")
        d_value = self._extract_number(governance_report, r"D 值約 (\d+(?:\.\d+)?)")
        latest_close = self._extract_number(governance_report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        macd_line = self._extract_number(governance_report, r"MACD 線約 (-?\d+(?:\.\d+)?)")
        signal_line = self._extract_number(governance_report, r"Signal 線約 (-?\d+(?:\.\d+)?)")
        histogram = self._extract_number(governance_report, r"Histogram 約 (-?\d+(?:\.\d+)?)")
        bollinger_position = self._extract_text(
            governance_report,
            r"(接近布林上軌|接近布林下軌|位於布林中軌偏上|位於布林中軌偏下|位於中軌附近)",
        )
        macd_trend = self._extract_text(governance_report, r"MACD 動能([偏多偏空轉強轉弱]+)")
        ma5_bias = self._extract_number(governance_report, r"MA5 乖離率約 (-?\d+(?:\.\d+)?)%")
        ma20_bias = self._extract_number(governance_report, r"MA20 乖離率約 (-?\d+(?:\.\d+)?)%")
        overbought = self._extract_text(governance_report, r"(尚未進入超買區|疑似進入超買區|已進入超買區)")
        if rsi and k_value and d_value and overbought and macd_trend and bollinger_position and ma5_bias and ma20_bias:
            return (
                f"{label} 最新技術指標顯示，RSI14 約 {rsi}，"
                f"K 值約 {k_value}，D 值約 {d_value}，{overbought}；"
                f"MACD 動能{macd_trend}，股價{bollinger_position}，"
                f"MA5 乖離率約 {ma5_bias}%，MA20 乖離率約 {ma20_bias}%。"
            )
        if latest_close and macd_line and signal_line and histogram:
            return (
                f"{label} 目前最新收盤價約 {latest_close} 元，"
                f"MACD 線約 {macd_line}，Signal 線約 {signal_line}，Histogram 約 {histogram}，"
                "但仍需結合 RSI、KD 與布林通道承接足夠證據後才能完整判讀。"
            )
        if latest_close:
            return f"{label} 目前已有最新價格資料，但不足以完整確認 RSI、KD、MACD、布林通道與均線乖離的綜合判讀。"
        return f"資料不足，無法確認{label}目前的 RSI、KD、MACD、布林通道與均線乖離狀態。"

    def _summarize_season_line_margin(self, label: str, governance_report: GovernanceReport) -> str:
        latest_close = self._extract_number(governance_report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        season_line = self._extract_number(governance_report, r"季線\(MA60\)約 (\d+(?:\.\d+)?) 元")
        season_line_status = self._extract_text(
            governance_report, r"(近期跌破季線|仍在季線下方|重新站回季線|尚未跌破季線)"
        )
        margin_balance = self._extract_number(governance_report, r"最新融資餘額約 (\d+(?:\.\d+)?) 張")
        utilization_pct = self._extract_number(governance_report, r"融資使用率約 (\d+(?:\.\d+)?)%")
        average_delta_pct = self._extract_number(governance_report, r"相較近 20 日平均變動約 (-?\d+(?:\.\d+)?)%")
        margin_status = self._extract_text(governance_report, r"籌碼面屬(偏高|中性偏高|中性偏低|中性)")
        if latest_close and season_line and season_line_status and margin_balance and utilization_pct and margin_status:
            delta_segment = f"，相較近 20 日平均變動約 {average_delta_pct}%" if average_delta_pct is not None else ""
            return (
                f"{label} 最新收盤價約 {latest_close} 元，季線約 {season_line} 元，"
                f"目前{season_line_status}；最新融資餘額約 {margin_balance} 張，"
                f"融資使用率約 {utilization_pct}%{delta_segment}，若以籌碼面推估屬{margin_status}。"
            )
        if latest_close and season_line and season_line_status:
            return (
                f"{label} 最新收盤價約 {latest_close} 元，季線約 {season_line} 元，"
                f"目前{season_line_status}，但融資餘額證據仍不足以完整評估市場看法。"
            )
        return f"資料不足，無法確認{label}股價是否跌破季線及融資餘額的市場觀感。"

    def _summarize_shipping(self, label: str, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        shipping_label = label if not comparison_label else f"{label}與{comparison_label}"
        has_red_sea = self._evidence_contains(governance_report, ("紅海", "紅海航線", "繞道", "受阻"))
        has_scfi = self._evidence_contains(governance_report, ("SCFI", "運價指數", "運價"))
        has_target_price = self._evidence_contains(governance_report, ("目標價", "評等", "分析師", "法人", "外資"))
        has_divergent_target = self._evidence_contains(governance_report, ("分歧", "正負解讀並存", "方向尚未完全一致"))
        has_positive_target = self._evidence_contains(governance_report, ("上修", "調高", "買進", "看好", "受惠", "有戲"))
        has_negative_target = self._evidence_contains(governance_report, ("下修", "調降", "保守", "觀望", "中立", "壓力"))
        if governance_report.evidence:
            points: list[str] = []
            if has_red_sea and has_scfi:
                points.append("紅海航線受阻與 SCFI 反彈，仍被市場視為短線運價的重要支撐來源")
            elif has_red_sea:
                points.append("紅海航線受阻仍被視為短線運價的支撐因子")
            elif has_scfi:
                points.append("現有報導多把 SCFI 與現貨運價變化當作短線評價核心")
            if has_target_price:
                if has_divergent_target or (has_positive_target and has_negative_target):
                    points.append("法人與分析師對目標價調整仍偏分歧")
                elif has_positive_target:
                    points.append("法人與分析師反應偏正向，目標價解讀較偏上修")
                elif has_negative_target:
                    points.append("法人與分析師反應偏保守，目標價解讀較偏下修")
                else:
                    points.append("法人報導多圍繞目標價與評等調整，但方向尚未完全一致")
            else:
                points.append("目前仍缺少足夠的目標價調整報導")
            if points:
                return f"{shipping_label} 目前可整理到的最新訊息顯示，" + "；".join(points) + "；後續仍要看 SCFI 續航與航線壅塞是否延續。"
            return f"{shipping_label} 目前已有部分航運事件新聞可供整理，但公開資訊仍不足以穩定判斷 SCFI 支撐力道與目標價調整方向。"
        return f"資料不足，無法確認紅海航線受阻對{shipping_label}的 SCFI 支撐力道與分析師目標價調整。"

    def _summarize_electricity(self, label: str, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        electricity_label = label if not comparison_label else f"{label}與{comparison_label}"
        has_tariff = self._evidence_contains(governance_report, ("工業電價", "電價", "調漲", "漲價", "電費"))
        has_cost = self._evidence_contains(governance_report, ("成本", "毛利", "壓力", "費用", "獲利"))
        has_amount = self._evidence_contains(governance_report, ("億元", "千萬元", "增加額度", "增幅", "%"))
        has_response = self._evidence_contains(
            governance_report, ("因應", "對策", "節能", "節電", "降耗", "轉嫁", "調價", "綠電", "自發電")
        )
        if governance_report.evidence:
            points: list[str] = []
            if has_tariff and has_cost:
                points.append("工業電價調漲仍被市場視為用電大戶的成本壓力來源")
            elif has_tariff:
                points.append("工業電價調漲是近期需要關注的成本變數")
            if has_amount:
                points.append("現有報導對成本增加額度已有部分估算，但仍需以公司正式說明為準")
            else:
                points.append("目前公開資訊多停留在方向性壓力描述，單一公司可精算的增加額度仍不完整")
            if has_response:
                points.append("公司常見因應方向多圍繞節能降耗、售價轉嫁與能源配置調整")
            else:
                points.append("目前對具體因應對策的揭露仍偏有限")
            return f"{electricity_label} 目前可整理到的最新訊息顯示，" + "；".join(points) + "。"
        return f"資料不足，無法確認工業電價調漲對{electricity_label}的成本增加幅度與因應對策。"

    def _summarize_macro_yield(self, label: str, governance_report: GovernanceReport) -> str:
        has_cpi = self._evidence_contains(governance_report, ("CPI", "通膨"))
        has_rate = self._evidence_contains(governance_report, ("利率", "美債", "殖利率", "高殖利率"))
        has_financial_sector = self._evidence_contains(governance_report, ("金控", "金控股", "金融股"))
        has_negative = self._evidence_contains(governance_report, ("負面", "保守", "觀望", "壓力", "降溫"))
        has_defensive = self._evidence_contains(governance_report, ("防禦", "現金流", "穩健", "支撐"))
        has_institutional_view = self._evidence_contains(governance_report, ("法人", "外資", "觀點", "看法"))
        if governance_report.evidence:
            points: list[str] = []
            if has_cpi and has_rate:
                points.append("美國 CPI 若偏熱，市場通常會把它解讀為利率下修延後或債息上行壓力，進而壓抑高殖利率股情緒")
            elif has_cpi:
                points.append("市場把 CPI 變化視為高殖利率標的的重要情緒變數")
            if has_financial_sector:
                points.append("這種情緒也常延伸到金控股與其他防禦型高殖利率標的")
            if has_institutional_view:
                if has_negative and has_defensive:
                    points.append("法人最新觀點偏向保守，但仍會提到現金流與防禦性支撐")
                elif has_negative:
                    points.append("法人最新觀點偏保守或觀望")
                elif has_defensive:
                    points.append("法人仍看重其防禦性與現金流穩定度")
                else:
                    points.append("法人觀點仍在觀察利率與殖利率評價變化")
            if points:
                if not has_cpi:
                    points.insert(0, "現有來源對美國 CPI 的直接傳導證據仍有限")
                return f"{label} 目前可整理到的最新訊息顯示，" + "；".join(points) + "。"
            return f"{label} 目前已有部分總經與高殖利率題材新聞可供整理，但仍不足以穩定判斷 CPI 對高殖利率股與金控股情緒的傳導幅度。"
        return f"資料不足，無法確認美國 CPI 對{label}與高殖利率族群的負面情緒影響與法人觀點。"

    def _summarize_theme_impact(self, label: str, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        theme_label = label if not comparison_label else f"{label}、{comparison_label}"
        has_cobalt_signal = self._evidence_contains(governance_report, ("鈷價", "鈷", "油價", "原料", "cobalt"))
        has_energy_transition = self._evidence_contains(governance_report, ("能源轉型", "城市採礦"))
        has_product_mix = self._evidence_contains(governance_report, ("產品組合", "庫存效應", "產能利用率"))
        has_asml_signal = self._evidence_contains(
            governance_report, ("ASML", "艾司摩爾", "展望", "不如預期", "保守", "下修")
        )
        has_equipment_signal = self._evidence_contains(
            governance_report, ("半導體設備", "設備", "EUV", "曝光機", "資本支出", "擴產", "訂單")
        )
        has_sentiment_signal = self._evidence_contains(
            governance_report, ("利空", "情緒", "觀望", "保守", "降溫", "下修")
        )
        if governance_report.evidence:
            points: list[str] = []
            if has_asml_signal:
                points.append("ASML 展望偏保守的訊號，容易被市場解讀為設備鏈短線逆風")
            if has_equipment_signal:
                points.append("半導體設備族群的觀察重點，會回到晶圓廠資本支出、擴產節奏與訂單能見度")
            if has_sentiment_signal:
                points.append("短線情緒偏向保守或觀望，利空解讀通常會先反映在題材熱度與估值")
            if has_energy_transition:
                points.append("近期公開資訊仍偏向能源轉型與材料題材")
            if has_product_mix:
                points.append("短線訊號仍以產品組合、庫存效應與產能利用率為主")
            if has_cobalt_signal:
                points.append("原料價格變動仍是影響波動的關鍵")
            if points:
                return f"{theme_label} 目前可整理到的最新訊息顯示，" + "；".join(points) + "。"
            return f"{theme_label} 目前有部分新聞可供參考，但公開資訊仍不足以直接量化 ASML 展望變化對設備族群基本面的實際傳導強度。"
        return f"資料不足，無法確認ASML展望轉弱對{theme_label}的最新利空分析與情緒影響。"

    def _summarize_dividend_yield(self, label: str, governance_report: GovernanceReport) -> str:
        cash_dividend = self._extract_number(governance_report, r"現金股利(?:合計)?約 (\d+(?:\.\d+)?) 元")
        close_price = self._extract_number(governance_report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        yield_pct = self._extract_number(governance_report, r"現金殖利率約 (\d+(?:\.\d+)?)%")
        if cash_dividend and close_price and yield_pct:
            return f"{label} 最新可取得的現金股利約 {cash_dividend} 元；若以最新收盤價約 {close_price} 元換算，現金殖利率約 {yield_pct}%。"
        if cash_dividend:
            return f"{label} 最新可取得的現金股利約 {cash_dividend} 元，但目前不足以穩定換算殖利率。"
        return f"{label} 目前已有部分股利資料，但不足以完整確認最新配息政策與現金殖利率。"

    def _summarize_ex_dividend_performance(self, label: str, governance_report: GovernanceReport) -> str:
        cash_dividend = self._extract_number(governance_report, r"現金股利約 (\d+(?:\.\d+)?) 元")
        ex_date = self._extract_text(governance_report, r"除息交易日為 (\d{4}-\d{2}-\d{2})")
        intraday_fill_ratio = self._extract_number(governance_report, r"盤中最高填息率約 (\d+(?:\.\d+)?)%")
        close_fill_ratio = self._extract_number(governance_report, r"收盤填息率約 (\d+(?:\.\d+)?)%")
        reaction = self._extract_text(governance_report, r"市場反應偏([中性正向強勁弱勢]+)")
        same_day_fill = self._extract_text(governance_report, r"(當天(?:盤中)?(?:完成|未完成)填息)")
        if ex_date and intraday_fill_ratio and close_fill_ratio:
            fill_sentence = f"除息日 {ex_date} 盤中最高填息率約 {intraday_fill_ratio}%，收盤填息率約 {close_fill_ratio}%。"
            if same_day_fill:
                fill_sentence += same_day_fill + "。"
            if reaction:
                fill_sentence += f"市場反應偏{reaction}。"
            return f"{label} 最新一次除息事件顯示，{fill_sentence}"
        if cash_dividend and ex_date:
            return f"{label} 目前可確認最新現金股利約 {cash_dividend} 元，除息日為 {ex_date}，但仍缺少足夠價格證據來確認當天的填息表現。"
        return f"資料不足，無法確認{label}除權息當天的填息表現與市場反應。"

    def _summarize_earnings_eps(self, label: str, governance_report: GovernanceReport) -> str:
        annual_eps = self._extract_number(governance_report, r"全年 EPS 約 (\d+(?:\.\d+)?) 元")
        latest_quarter_eps = self._extract_number(governance_report, r"最新一季 EPS 約 (\d+(?:\.\d+)?) 元")
        cash_dividend = self._extract_number(governance_report, r"現金股利(?:合計)?約 (\d+(?:\.\d+)?) 元")
        if annual_eps and cash_dividend:
            return f"{label} 去年全年 EPS 約 {annual_eps} 元；目前已取得的股利資料顯示，現金股利約 {cash_dividend} 元，市場預期仍需配合最新公告與新聞觀察。"
        if latest_quarter_eps and cash_dividend:
            return f"{label} 最新一季 EPS 約 {latest_quarter_eps} 元；目前可取得的股利資料顯示，現金股利約 {cash_dividend} 元。"
        if annual_eps:
            return f"{label} 去年全年 EPS 約 {annual_eps} 元，但股利預期仍需等待更多公告或新聞佐證。"
        return f"{label} 目前已有部分財報或股利資料，但仍不足以完整確認 EPS 與市場股利預期。"

    # ================================================================== #
    # IMPACTS builders — one per Intent                                   #
    # ================================================================== #

    def _build_impacts_news_digest(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.SHIPPING.value in tags or "SCFI" in tags:
            return [
                "紅海航線受阻與 SCFI 變化，常是貨櫃航運股短線評價最直接的事件指標。",
                "把運價支撐與分析師目標價一起看，比單看題材新聞更能分辨市場是在交易基本面，還是在交易情緒。",
                "若長榮與陽明都被同一波航運新聞反覆提及，通常代表市場正在把它們視為同一組運價受惠標的。",
            ]
        if TopicTag.ELECTRICITY.value in tags:
            return [
                "工業電價調漲通常會先影響高耗電產業的成本結構，再慢慢反映到毛利率與報價策略。",
                "把成本壓力和公司因應對策一起看，比只看漲價新聞更能判斷衝擊是短期還是可被吸收。",
                "若中鋼、台泥這類用電大戶都被同一波電價新聞反覆點名，通常代表市場正在重新評估其成本轉嫁能力。",
            ]
        if TopicTag.MACRO.value in tags or "殖利率" in tags or "CPI" in tags:
            return [
                "美國 CPI 與通膨預期，常會先透過利率與債息路徑影響高殖利率股的評價。",
                "把高殖利率個股與金控股放在一起看，可以更快分辨市場是在交易殖利率吸引力，還是在交易利率風險。",
                "法人觀點若同時提到防禦性與壓力，通常代表市場並非全面看空，而是在重新校正估值。",
            ]
        if TopicTag.GUIDANCE.value in tags or "指引" in tags:
            return [
                "法說後的媒體與法人反應，可協助觀察市場如何解讀公司對下半年需求、毛利與出貨節奏的說法。",
                "若正面與負面反應同時存在，通常代表市場對指引的可持續性仍有分歧。",
                "比起單一標題，交叉比對多家媒體與法人觀點，更能看出真正的焦點與疑慮。",
            ]
        if TopicTag.LISTING.value in tags:
            return [
                "轉上市後的股價波動常同時受籌碼、題材與營運消息影響，不能只用單一新聞解釋。",
                "月營收若明顯年增，通常代表需求或航線布局有支撐，但不等於獲利會同步放大。",
                "若市場消息主要集中在品牌、票價或新航線題材，股價反應也可能比基本面更快。",
            ]
        if "題材" in tags or "產業" in tags:
            return [
                "主題型問題要先確認有沒有公開資訊直接提到該傳導關係，不適合只靠產業想像補空。",
                "對半導體設備族群來說，短線更常見的觀察點會回到晶圓廠資本支出、擴產節奏與訂單能見度。",
                "若沒有更直接的訂單、出貨或公司說明，對「展望不如預期」的傳導幅度應保持低確信度。",
            ]
        # 通用
        return [
            "新聞能幫助理解市場目前關注的事件與敘事。",
            "若多個來源集中報導同一主題，代表關注度可能升高。",
            "仍需回看原文與公告，避免只看摘要造成誤讀。",
        ]

    def _build_impacts_earnings_review(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.GROSS_MARGIN.value in tags or "轉正" in tags:
            return [
                "毛利率是否由負轉正，能快速判斷產品組合、報價與成本吸收能力是否有回到健康區間。",
                "營業利益是否同步轉正，比單看毛利率更能判斷本業是否真的擺脫虧損。",
                "若毛利率改善但營業利益仍為負，通常代表費用、折舊或稼動率壓力仍在，本業復甦未必已站穩。",
            ]
        if TopicTag.REVENUE.value in tags or "月增率" in tags or "年增率" in tags:
            return [
                "月營收累計年增率可以快速觀察公司當年開局的營運動能。",
                "拿今年前幾個月與去年同期比較，能避免單看單月數字受基期或出貨時點影響。",
                "營收年增可以當作基本面熱度的先行指標，但仍需搭配毛利率與 EPS 才能完整解讀。",
            ]
        return [
            "EPS 表現可反映公司獲利能力與資本支出承受度。",
            "股利資訊常影響市場對現金回饋與殖利率的預期。",
            "若財報與股利方向一致，通常更容易形成穩定敘事。",
        ]

    def _build_impacts_valuation_check(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if is_fundamental_valuation_question(query):
            return [
                "基本面可以幫助判斷獲利與營運動能是否具備延續性。",
                "本益比則可以拿來定位市場目前給這檔公司的估值水位。",
                "若要做進場判斷，最好是把獲利趨勢和估值位置一起看。",
            ]
        if "股價區間" in tags:
            return [
                "區間高低點可用來觀察短線波動與市場交易熱度。",
                "若價格靠近區間上緣或下緣，需搭配後續消息確認是否延續。",
                "單看區間無法判斷基本面是否同步改善或轉弱。",
            ]
        return [
            "本益比能幫助估計市場目前願意用多高的評價倍數反映公司獲利能力。",
            "把目前本益比放回近 13 個月歷史區間，比單看單一倍數更容易判斷估值是否偏高或偏低。",
            "對長期投資來說，本益比只是進場價格的一個觀察面，仍需搭配成長性與現金流一起看。",
        ]

    def _build_impacts_dividend_analysis(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return [
                "自由現金流可以幫助觀察公司在維持資本支出之後，還剩下多少現金能支應股利或其他資金用途。",
                "把現金股利發放總額與 FCF 一起看，比單看殖利率更能判斷股利政策是否有現金流支撐。",
                "若連續多年 FCF 高於現金股利支出，通常代表現階段股利政策較具穩定基礎。",
            ]
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return [
                "負債比率可以協助觀察公司近幾期的槓桿水位是否突然墊高。",
                "把最新負債比率和前一季、去年同期一起比較，較能分辨是短期波動還是財務結構轉弱。",
                "現金及約當現金若明顯高於現金股利總額，通常代表眼前的股利支付緩衝仍在。",
            ]
        if "除息" in tags or "填息" in tags:
            return [
                "除權息當天的填息率可以反映市場對股利題材的接受度。",
                "盤中最高填息率與收盤填息率可協助區分是短線衡動還是買盤延續。",
                "交易量與當日價格反應一起看，比單看漲跌更能描述市場態度。",
            ]
        return [
            "股利政策可反映公司對現金回饋的安排。",
            "現金殖利率可協助估算以目前股價換算的現金回饋水準。",
            "殖利率仍會隨股價變動，不是固定值。",
        ]

    def _build_impacts_financial_health(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if query.comparison_ticker or query.comparison_company_name:
            return [
                "毛利率較高的一方，通常代表定價能力、貨源結構或成本控管相對較占優勢。",
                "若兩家公司的毛利率差距能在近幾季持續，較容易反映為結構性差異，不只是單季波動。",
                "不過「經營效率」不能只看毛利率，仍要搭配營益率、費用率和資產週轉一起觀察。",
            ]
        if "獲利" in tags or "穩定性" in tags:
            return [
                "拿近五年的年度獲利一起看，比單看最新一年更能判斷這檔股票是否適合當成長期現金流型部位觀察。",
                "若近年出現轉虧，就不適合直接視為「穩定獲利」公司，後續要再看是週期性還是結構性問題。",
                "若處於轉虧年，能再拆出是本業轉弱還是業外拖累，會比只看 EPS 更有參考價值。",
            ]
        if "成長" in tags or "AI" in tags:
            return [
                "AI 伺服器營收占比屬於基本面結構資訊，應優先看公司說法、法說與高可信新聞整理。",
                "2026 年成長預測通常來自管理層展望、產能擴張與產業需求判讀，不應直接當成已實現結果。",
                "若只有成長敘事而沒有一致數字，較適合解讀為方向性訊號，而不是精確預估。",
            ]
        return self._build_impacts_generic()

    def _build_impacts_technical_view(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return [
                "季線是中短期趨勢的常見觀察線，能快速建立目前價格強弱的背景。",
                "融資餘額與融資使用率能協助觀察槓桿資金是否集中，對短線波動通常比較敏感。",
                "把季線位置與融資餘額一起看，比單看價格或單看籌碼更容易掌握風險。",
            ]
        return [
            "RSI 與 KD 能幫助觀察短期買盤熱度與強弱變化。",
            "MACD 適合用來看價格動能是否持續轉強或轉弱，布林通道則可以觀察股價相對區間位置。",
            "均線乖離可用來檢查短線漲勢是否過熱或過冷，搭配多個指標會比單看 RSI 或 KD 更完整。",
        ]

    def _build_impacts_investment_assessment(self, query: StructuredQuery) -> list[str]:
        if is_fundamental_valuation_question(query):
            return [
                "基本面可以幫助判斷獲利與營運動能是否具備延續性。",
                "本益比則可以拿來定位市場目前給這檔公司的估值水位。",
                "若要做進場判斷，最好是把獲利趨勢和估值位置一起看。",
            ]
        tags = set(query.topic_tags)
        if "公告" in tags:
            return [
                "公告通常會影響市場對公司短期事件與決策節奏的理解。",
                "若公告涉及股利、董事會或法說，可能提高後續關注度。",
                "公告本身仍需搭配財報與新聞交叉驗證，避免過度解讀。",
            ]
        return [
            "短線漲跌通常同時受到基本面、資金面與消息面影響。",
            "若近期有公告或財報更新，市場反應可能放大波動。",
            "缺少多來源佐證時，任何方向判斷都應保守看待。",
        ]

    def _build_impacts_generic(self) -> list[str]:
        return [
            "現有資料可作為快速掌握脈絡的起點。",
            "若不同來源能互相印證，判讀穩定度會更高。",
            "後續仍應持續追蹤新的公告、財報與新聞變化。",
        ]

    # ================================================================== #
    # RISKS builders — one per Intent                                     #
    # ================================================================== #

    def _build_risks_news_digest(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.SHIPPING.value in tags or "SCFI" in tags:
            return [
                "SCFI 與紅海事件多屬短線催化，未必能直接等同全年獲利或長期運價中樞上移。",
                "分析師目標價調整常受事件時點與市場情緒影響，方向可能很快反轉。",
                "若後續紅海繞道、塞港或運力供給狀況緩解，短線支撐力道也可能同步降溫。",
            ]
        if TopicTag.ELECTRICITY.value in tags:
            return [
                "電價調漲帶來的成本壓力不一定能完整轉嫁，對毛利率的影響仍要看產品報價與景氣狀況。",
                "若公開資訊沒有揭露可精算的用電成本結構，單一公司增加額度多半只能做方向性判讀。",
                "公司宣示的節能或因應方案，未必能在短期內完全抵消漲價衝擊。",
            ]
        if TopicTag.MACRO.value in tags or "殖利率" in tags or "CPI" in tags:
            return [
                "CPI 對高殖利率股的影響常先反映在評價與情緒面，不一定立刻反映到基本面。",
                "若市場很快把焦點轉回降息時點或債息回落，高殖利率股的負面情緒也可能快速修正。",
                "法人觀點常同時包含防禦性與估值壓力，若只看單一方向容易過度解讀。",
            ]
        if TopicTag.GUIDANCE.value in tags or "指引" in tags:
            return [
                "法說後的媒體與法人解讀常帶有主觀判斷，未必等同公司最終實際營運結果。",
                "若只有少數報導提到下半年指引，市場情緒可能被單一敘事放大。",
                "即使會後反應偏正面，仍需追蹤後續月營收、財測與客戶拉貨節奏是否驗證。",
            ]
        if TopicTag.LISTING.value in tags:
            return [
                "轉上市初期的價格波動可能夾雜籌碼與情緒因素，未必完全反映中長期基本面。",
                "單月營收年增或月增偏強，仍可能受旺季、航線擴張或一次性因素影響。",
                "若缺少公司法說或更直接公告佐證，對股價波動主因的解讀仍應保守。",
            ]
        if "題材" in tags or "產業" in tags:
            return [
                "主題與個股關聯不代表會立即傳導成營收或獲利變化。",
                "若短線新聞主要在談供應鏈展望、資本支出或題材情緒，不一定能直接代表最終訂單真實轉弱。",
                "缺少公司公告、訂單或法說證據時，對影響幅度的判斷應控制在描述層級。",
            ]
        return self._build_risks_generic(query)

    def _build_risks_earnings_review(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.GROSS_MARGIN.value in tags or "轉正" in tags:
            return [
                "毛利率轉正不一定等於整體獲利體質已經穩定改善，仍要看營業利益是否同步回正。",
                "若營業利益仍為負，代表費用結構、折舊負擔或稼動率壓力可能還在，本業復甦未必已站穩。",
                "單一季度的轉正也可能受匯率、產品組合或一次性因素影響，還需要連續幾季追蹤。",
            ]
        if TopicTag.REVENUE.value in tags or "月增率" in tags:
            return [
                "累計營收年增只反映營收端的變化，不直接代表毛利率或獲利同步改善。",
                "單看前幾個月的累計營收，仍可能受出貨時點、匯率或季節性因素影響。",
                "若沒有同時搭配財報與法說，容易過度把營收成長解讀為獲利確定性。",
            ]
        return [
            "股利最終內容仍需以董事會、股東會或公司正式公告為準。",
            "歷史 EPS 不能直接保證未來獲利延續。",
            "若市場預期主要來自單一新聞來源，解讀上要保留彈性。",
        ]

    def _build_risks_valuation_check(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if is_fundamental_valuation_question(query):
            return [
                "估值高低會隨股價與獲利預期變化，今天的本益比位置不代表之後不會再調整。",
                "基本面如果只有單一季或單月證據，不一定能代表中長期獲利趨勢。",
                "單看本益比或單看營收都可能失真，仍要搭配現金流、資本支出和產業景氣一起解讀。",
            ]
        if is_forward_price_question(query):
            return [
                "漲跌預測本身具有高度不確定性，不能視為確定方向。",
                "短期價格可能受到外部消息、資金輪動與整體市場風險影響。",
                "若近期有公告或財報更新，市場反應可能放大波動。",
            ]
        if "股價區間" in tags:
            return [
                "價格區間只反映歷史交易結果，不能直接推論未來方向。",
                "若期間內剛好出現重大事件，區間可能失真放大短期波動。",
                "未結合基本面與公告時，容易忽略趨勢反轉風險。",
            ]
        return [
            "本益比會隨股價與獲利預期變動，今天的估值位置不代表之後不會再調整。",
            "歷史偏高不一定代表馬上貴到不能買，仍需搭配未來 EPS 成長與產業景氣判斷。",
            "若只用本益比決定是否進場，可能忽略現金流、股利與資本支出等更適合長投的變數。",
        ]

    def _build_risks_dividend_analysis(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return [
                "自由現金流雖能說明現金支應能力，仍不代表未來股利一定維持不變。",
                "若後續資本支出提升、現金流轉弱或監理政策改變，股利政策仍可能調整。",
                "現金股利發放總額為推估值，仍應以公司正式公告與實際發放安排為準。",
            ]
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return [
                "負債比率短期變動不一定代表財務體質立即惡化，仍需拆解是應付帳款、借款或營運週轉造成。",
                "帳上現金高於股利總額，只能說目前支付緩衝存在，仍需搭配未來現金流與資本支出一起看。",
                "若後續公司調整股利政策、擴大投資或出現一次性資金需求，現有支應能力判斷也可能改變。",
            ]
        if "除息" in tags or "填息" in tags:
            return [
                "填息表現只反映當天市場行為，不代表後續走勢一定延續。",
                "若單日波動受到大盤或外部消息影響，可能放大或扭曲填息觀察。",
                "若缺少完整交易資料或公開報導，對市場反應的描述應保守解讀。",
            ]
        return [
            "股利最終內容仍需以董事會、股東會或公司正式公告為準。",
            "現金殖利率會隨股價變動，查詢當下與實際買入時點可能不同。",
            "若公司後續更新股利政策，現有換算結果需要重新檢查。",
        ]

    def _build_risks_financial_health(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if query.comparison_ticker or query.comparison_company_name:
            return [
                "毛利率較高不一定代表整體經營效率就更好，仍要看營益率、費用控管與資產週轉。",
                "航運股毛利率容易受運價週期、燃油成本與航線結構影響，單一季數字可能波動很大。",
                "若比較時點不完全一致，或財報口徑剛好受一次性因素影響，解讀上要保守。",
            ]
        if "獲利" in tags or "穩定性" in tags:
            return [
                "就算近幾年都有獲利，若獲利幅度起伏很大，對「退休存股」的穩定性仍要保守看待。",
                "轉虧年的原因若只能從財報結構推估，仍應與公司法說或年報說明互相對照。",
                "傳統產業獲利容易受景氣、原料報價和利差影響，過去五年不一定能直接代表未來五年。",
            ]
        if "成長" in tags or "AI" in tags:
            return [
                "營收占比若不是公司正式揭露，常來自媒體或法人估算，可能隨季度而變動。",
                "2026 年成長預測具有前瞻不確定性，容易受雲端資本支出、客戶拉貨節奏與供應鏈瓶頸影響。",
                "若現有證據主要是新聞敘事而非公司明確數字，對成長幅度的解讀應保守。",
            ]
        return self._build_risks_generic(query)

    def _build_risks_technical_view(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return [
                "跌破季線不一定代表中期趨勢確認轉空，仍需觀察後續幾個交易日是否繼續失守。",
                "融資餘額偏高不代表必然會出現修正，但對波動放大的風險確實較高。",
                "若公開資料缺少主流來源直接評論融資熱度，對「市場看法」的描述仍應以籌碼推估為主。",
            ]
        return [
            "超買不代表股價會立即回檔，仍可能因強勢趨勢持續上行。",
            "技術指標是根據歷史價格推算，遇到突發消息時可能很快失真。",
            "若 MACD 與 RSI、KD 訊號不一致，表示短線動能與價格位階可能正在拉鋸，不宜單看單一指標下結論。",
        ]

    def _build_risks_investment_assessment(self, query: StructuredQuery) -> list[str]:
        if is_fundamental_valuation_question(query):
            return [
                "估值高低會隨股價與獲利預期變化，今天的本益比位置不代表之後不會再調整。",
                "基本面如果只有單一季或單月證據，不一定能代表中長期獲利趨勢。",
                "單看本益比或單看營收都可能失真，仍要搭配現金流、資本支出和產業景氣一起解讀。",
            ]
        tags = set(query.topic_tags)
        if "公告" in tags:
            return [
                "公告標題容易被快速解讀，仍需回看完整揭露內容。",
                "若只有單一公告而缺乏交叉來源，市場解讀可能偏差。",
                "公告對股價的影響常受時間點與市場情緒放大或鈍化。",
            ]
        return self._build_risks_generic(query)

    def _build_risks_generic(self, query: StructuredQuery) -> list[str]:
        if query.stance_bias != StanceBias.NEUTRAL:
            return [
                "當問題帶有單一立場時，容易忽略相反證據。",
                "若只挑選支持既有看法的資訊，判讀偏誤會被放大。",
                "投資判斷仍應同時檢查利多、利空與不確定因素。",
            ]
        return [
            "若資料主要集中在單一時間窗，可能忽略更長期的基本面變化。",
            "若後續出現更新公告，現有摘要需要重新驗證。",
            "不同來源對同一事件的解讀可能不同，需持續比對。",
        ]

    # ================================================================== #
    # Helper utilities (unchanged)                                        #
    # ================================================================== #

    def _extract_price_range(self, governance_report: GovernanceReport) -> tuple[str | None, str | None]:
        high_pattern = re.compile(r"最高價(?:為)?\s*(\d+(?:\.\d+)?)\s*元")
        low_pattern = re.compile(r"最低價(?:為)?\s*(\d+(?:\.\d+)?)\s*元")
        for evidence in governance_report.evidence:
            high_match = high_pattern.search(evidence.excerpt)
            low_match = low_pattern.search(evidence.excerpt)
            if high_match and low_match:
                return high_match.group(1), low_match.group(1)
        return None, None

    def _extract_company_margin(self, governance_report: GovernanceReport, company_name: str) -> str | None:
        pattern = re.compile(rf"{re.escape(company_name)}[^。；]*?毛利率約 (\d+(?:\.\d+)?)%")
        for evidence in governance_report.evidence:
            match = pattern.search(evidence.excerpt)
            if match:
                return match.group(1)
        return None

    def _extract_year_metric_pairs(
        self,
        governance_report: GovernanceReport,
        pattern: str,
    ) -> list[tuple[str, str]]:
        compiled = re.compile(pattern)
        pairs: list[tuple[str, str]] = []
        seen_years: set[str] = set()
        for evidence in governance_report.evidence:
            match = compiled.search(evidence.excerpt)
            if not match:
                continue
            year = match.group(1)
            value = match.group(2)
            if year in seen_years:
                continue
            seen_years.add(year)
            pairs.append((year, value))
        pairs.sort(key=lambda item: item[0])
        return pairs

    def _extract_number(self, governance_report: GovernanceReport, pattern: str) -> str | None:
        compiled = re.compile(pattern)
        for evidence in governance_report.evidence:
            match = compiled.search(evidence.excerpt)
            if match:
                return match.group(1)
        return None

    def _extract_text(self, governance_report: GovernanceReport, pattern: str) -> str | None:
        compiled = re.compile(pattern)
        for evidence in governance_report.evidence:
            match = compiled.search(evidence.excerpt)
            if match:
                return match.group(1)
        return None

    def _evidence_contains(self, governance_report: GovernanceReport, keywords: tuple[str, ...]) -> bool:
        for evidence in governance_report.evidence:
            haystack = f"{evidence.title} {evidence.excerpt}".lower()
            if any(keyword.lower() in haystack for keyword in keywords):
                return True
        return False

    def _listing_event_label(self, query: StructuredQuery) -> str:
        lowered_query = query.user_query.lower()
        if "ipo" in lowered_query or "initial public offering" in lowered_query:
            return "IPO 後"
        if "掛牌" in query.user_query:
            return "掛牌後"
        return 