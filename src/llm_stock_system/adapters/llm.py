import re

from llm_stock_system.core.enums import SourceTier, StanceBias, Topic
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
    """Fallback synthesizer that keeps answers grounded in retrieved evidence."""

    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
    ) -> AnswerDraft:
        _ = system_prompt

        if not governance_report.evidence:
            return AnswerDraft(
                summary="\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d\u3002",
                highlights=["\u73fe\u6709\u8b49\u64da\u4e0d\u8db3\u6216\u4e00\u81f4\u6027\u4e0d\u8db3\uff0c\u7cfb\u7d71\u5df2\u964d\u7d1a\u56de\u7b54\u3002"],
                facts=["\u5c1a\u672a\u53d6\u5f97\u8db3\u5920\u7684\u5b98\u65b9\u516c\u544a\u3001\u65b0\u805e\u6216\u8ca1\u5831\u8cc7\u6599\u3002"],
                impacts=["\u8cc7\u6599\u4e0d\u8db3\u6642\uff0c\u4e0d\u61c9\u5c07\u55ae\u4e00\u8a0a\u606f\u76f4\u63a5\u89e3\u8b80\u70ba\u8da8\u52e2\u3002"],
                risks=[
                    "\u8cc7\u6599\u4e0d\u8db3\u6642\uff0c\u5bb9\u6613\u8aa4\u628a\u55ae\u4e00\u8a0a\u606f\u7576\u6210\u8da8\u52e2\u3002",
                    "\u82e5\u53ea\u4f9d\u8cf4\u672a\u9a57\u8b49\u8cc7\u8a0a\uff0c\u53ef\u80fd\u653e\u5927\u5224\u65b7\u504f\u8aa4\u3002",
                    "\u5efa\u8b70\u7b49\u5f85\u66f4\u591a\u516c\u544a\u3001\u8ca1\u5831\u6216\u4e3b\u6d41\u4f86\u6e90\u66f4\u65b0\u3002",
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

        highlights = [f"{item.title}\uff08{item.source_name}\uff09" for item in governance_report.evidence[:3]]
        facts = [
            f"{item.source_name} \u65bc {item.published_at:%Y-%m-%d} \u63d0\u4f9b\u8cc7\u6599\uff1a{item.excerpt}"
            for item in governance_report.evidence[:3]
        ]
        if is_fundamental_valuation_question(query):
            highlights = build_fundamental_valuation_highlights(query, governance_report)
            facts = build_fundamental_valuation_facts(query, governance_report)
        if query.question_type == "price_outlook" and is_forward_price_question(query):
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

    def _build_summary(self, query: StructuredQuery, governance_report: GovernanceReport) -> str:
        label = query.company_name or query.ticker or "\u6b64\u6a19\u7684"

        if query.question_type == "margin_turnaround_review":
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
                gross_margin_status_text = (
                    gross_margin_status if gross_margin_status.endswith("。") else f"{gross_margin_status}。"
                )
                operating_status_text = (
                    operating_status if operating_status.endswith("。") else f"{operating_status}。"
                )
                profitability_view_text = profitability_view or ""
                if profitability_view_text and not profitability_view_text.endswith("。"):
                    profitability_view_text = f"{profitability_view_text}。"
                summary = (
                    f"{label} 最新季毛利率約 {latest_margin}%，"
                    f"{gross_margin_status_text}"
                    f"營業利益約 {latest_operating_income} 億元，{operating_status_text}"
                )
                if previous_margin is not None and previous_operating_income is not None:
                    summary += (
                        f"上一季毛利率約 {previous_margin}%，"
                        f"營業利益約 {previous_operating_income} 億元。"
                    )
                if profitability_view_text:
                    summary += profitability_view_text
                return summary
            return f"資料不足，無法確認{label}最新季毛利率是否由負轉正，以及營業利益是否同步轉正。"

        if query.question_type == "listing_revenue_review":
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

        if query.question_type == "guidance_reaction_review":
            positive_count = self._extract_number(governance_report, r"正面解讀共有 (\d+) 則")
            negative_count = self._extract_number(governance_report, r"負面解讀共有 (\d+) 則")
            has_positive = positive_count is not None or self._evidence_contains(governance_report, ("正面反應整理", "看好", "調高", "上修"))
            has_negative = negative_count is not None or self._evidence_contains(governance_report, ("負面反應整理", "保守", "下修", "不如預期"))
            if has_positive and has_negative:
                return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正負並存；目前可整理到同時存在正面與負面反應，顯示市場對後續動能仍有分歧。"
            if has_positive:
                return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正面；現有中高可信來源較多聚焦於成長動能或展望支撐。"
            if has_negative:
                return f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向保守；現有來源較多聚焦於下修、壓力或不確定性。"
            return f"資料不足，無法確認{label}法說會後媒體與法人對下半年營運指引的正負面反應。"

        if query.question_type == "monthly_revenue_yoy_review":
            availability_sentence = self._extract_text(
                governance_report,
                r"(截至 \d{4}-\d{2}-\d{2}，官方月營收資料最新僅到 \d{4}-\d{2}，尚未公布 \d{4}-\d{2} 月營收。[^。]*。)",
            )
            if availability_sentence:
                return availability_sentence

            month_revenue = self._extract_number(governance_report, r"單月營收約 (\d+(?:\.\d+)?) 億元")
            mom_pct = self._extract_number(governance_report, r"月增率約 (-?\d+(?:\.\d+)?)%")
            yoy_pct = self._extract_number(governance_report, r"年增率約 (-?\d+(?:\.\d+)?)%")
            mom_status = self._extract_text(
                governance_report,
                r"(月增率已超過 20%|月增率未達 20%)",
            )
            high_status = self._extract_text(
                governance_report,
                r"(創下近一年新高|尚未創下近一年新高|近一年月營收歷史不足[^。]*。)",
            )
            market_view = self._extract_text(
                governance_report,
                r"(市場解讀：[^。]*。)",
            )
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

        if query.question_type == "price_range":
            high_price, low_price = self._extract_price_range(governance_report)
            if high_price and low_price:
                return f"{label} \u8fd1 {query.time_range_days} \u5929\u8cc7\u6599\u986f\u793a\uff0c\u6700\u9ad8\u50f9\u70ba {high_price} \u5143\uff0c\u6700\u4f4e\u50f9\u70ba {low_price} \u5143\u3002"
            return f"{label} \u76ee\u524d\u5df2\u6709\u80a1\u50f9\u8cc7\u6599\uff0c\u4f46\u4e0d\u8db3\u4ee5\u7a69\u5b9a\u6574\u7406\u51fa\u5b8c\u6574\u5340\u9593\u3002"

        if query.question_type == "gross_margin_comparison_review":
            primary_label = query.company_name or query.ticker or "\u7b2c\u4e00\u5bb6\u516c\u53f8"
            comparison_label = query.comparison_company_name or query.comparison_ticker or "\u7b2c\u4e8c\u5bb6\u516c\u53f8"
            primary_margin = self._extract_company_margin(governance_report, primary_label)
            comparison_margin = self._extract_company_margin(governance_report, comparison_label)
            higher_company = self._extract_text(
                governance_report,
                rf"\u7531\s*({re.escape(primary_label)}|{re.escape(comparison_label)})\s*\u8f03\u9ad8",
            )
            margin_gap = self._extract_number(governance_report, r"\u9ad8\u51fa (\d+(?:\.\d+)?) \u500b\u767e\u5206\u9ede")
            if primary_margin and comparison_margin:
                resolved_higher_company = higher_company
                if resolved_higher_company is None:
                    resolved_higher_company = (
                        primary_label if float(primary_margin) >= float(comparison_margin) else comparison_label
                    )
                gap_segment = (
                    f"\uff0c\u9ad8\u51fa {margin_gap} \u500b\u767e\u5206\u9ede"
                    if margin_gap is not None
                    else ""
                )
                return (
                    f"\u82e5\u4ee5\u6700\u65b0\u53ef\u6bd4\u8ca1\u5831\u53e3\u5f91\u6bd4\u8f03\uff0c"
                    f"{primary_label} \u6bdb\u5229\u7387\u7d04 {primary_margin}%\uff0c"
                    f"{comparison_label} \u6bdb\u5229\u7387\u7d04 {comparison_margin}%\uff0c"
                    f"\u7531 {resolved_higher_company} \u8f03\u9ad8{gap_segment}\u3002"
                    f"\u55ae\u770b\u6bdb\u5229\u7387\uff0c{resolved_higher_company} \u5728\u5b9a\u50f9\u80fd\u529b\u6216\u6210\u672c\u7d50\u69cb\u4e0a\u76f8\u5c0d\u8f03\u4f54\u512a\u52e2\uff0c"
                    "\u4f46\u4e0d\u80fd\u55ae\u6191\u6bdb\u5229\u7387\u5c31\u7b49\u540c\u6574\u9ad4\u7d93\u71df\u6548\u7387\u66f4\u597d\u3002"
                )
            return (
                f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{primary_label}\u8207{comparison_label}"
                "\u6700\u65b0\u53ef\u6bd4\u8ca1\u5831\u53e3\u5f91\u4e0b\u7684\u6bdb\u5229\u7387\u5dee\u7570\u8207\u7d93\u71df\u6548\u7387\u8a55\u4f30\u3002"
            )

        if query.question_type == "profitability_stability_review":
            stability_sentence = self._extract_text(
                governance_report,
                r"(\u8fd1\u4e94\u5e74[^。]*(?:\u7a69\u5b9a\u7372\u5229|\u6ce2\u52d5[^。]*|\u4e26\u975e\u6bcf\u5e74\u90fd\u6709\u7a69\u5b9a\u7372\u5229)[^。]*。)",
            )
            loss_year = self._extract_number(governance_report, r"\u5728 (\d{4}) \u5e74\u6b78\u5c6c\u6bcd\u516c\u53f8\u6de8\u640d\u7d04")
            loss_amount = self._extract_number(governance_report, r"\u6de8\u640d\u7d04 (-?\d+(?:\.\d+)?) \u5104\u5143")
            loss_reason = self._extract_text(
                governance_report,
                r"(\u82e5\u53ea\u770b\u8ca1\u5831\u7d50\u69cb\u63a8\u4f30[^。]*。)",
            )
            if stability_sentence and loss_year and loss_amount:
                tail = f"{loss_reason}" if loss_reason else ""
                normalized_loss_amount = str(abs(float(loss_amount)))
                return f"{stability_sentence}{loss_year} 年是較明顯的虧損年度，歸屬母公司淨損約 {normalized_loss_amount} 億元。{tail}"
            if stability_sentence:
                return stability_sentence
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u904e\u53bb\u4e94\u5e74\u7684\u7372\u5229\u7a69\u5b9a\u6027\u8207\u8f49\u8667\u5e74\u5ea6\u539f\u56e0\u3002"

        if query.question_type == "monthly_revenue_yoy_review":
            month_count = self._extract_number(governance_report, r"\d{4} 年前 (\d+) 個月累計營收約")
            current_total = self._extract_number(governance_report, r"\d{4} 年前 \d+ 個月累計營收約 (\d+(?:\.\d+)?) 億元")
            previous_total = self._extract_number(governance_report, r"\d{4} 年同期約 (\d+(?:\.\d+)?) 億元")
            yoy_pct = self._extract_number(governance_report, r"年增率約 (-?\d+(?:\.\d+)?)%")
            availability_note = self._extract_text(governance_report, r"(截至 \d{4}-\d{2}，官方月營收資料僅更新到今年前 \d+ 個月。)")
            if current_total and previous_total and yoy_pct:
                leading = f"{availability_note}" if availability_note else ""
                month_label = month_count or "3"
                return f"{leading}{label} 今年前 {month_label} 個月累計營收約 {current_total} 億元，去年同期約 {previous_total} 億元，年增率約 {yoy_pct}%。"
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u4eca\u5e74\u524d\u4e09\u500b\u6708\u7d2f\u8a08\u71df\u6536\u8207\u53bb\u5e74\u540c\u671f\u7684\u6210\u9577\u5e45\u5ea6\u3002"

        if query.question_type == "debt_dividend_safety_review":
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
                payout_segment = f"；最新現金及約當現金約 {cash_balance} 億元，對最近一次現金股利總額約 {dividend_total} 億元，約可覆蓋 {coverage_ratio} 倍"
                payout_tail = f"，{payout_view}" if payout_view else ""
                return (
                    f"{label} 最新負債比率約 {debt_ratio}%{previous_segment}，{debt_status}"
                    f"{payout_segment}{payout_tail}。"
                )
            return f"資料不足，無法確認{label}負債比率是否突然升高，以及現金部位是否足以支應股利。"

        if query.question_type == "fcf_dividend_sustainability_review":
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
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u904e\u53bb\u4e09\u5e74\u7684\u81ea\u7531\u73fe\u91d1\u6d41\u3001\u73fe\u91d1\u80a1\u5229\u767c\u653e\u7e3d\u984d\u8207\u80a1\u5229\u653f\u7b56\u6c38\u7e8c\u6027\u3002"

        if query.question_type == "pe_valuation_review":
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
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u76ee\u524d\u672c\u76ca\u6bd4\u5728\u6b77\u53f2\u5340\u9593\u6240\u8655\u7684\u4f4d\u7f6e\u8207\u662f\u5426\u504f\u8cb4\u3002"

        if is_fundamental_valuation_question(query):
            return build_fundamental_valuation_summary(query, governance_report)

        if query.question_type == "price_outlook":
            return build_forward_price_summary(query, governance_report)

        if query.question_type == "revenue_growth_review":
            share_value = self._extract_number(governance_report, r"(\d+(?:\.\d+)?)%")
            has_ai_server = self._evidence_contains(governance_report, ("ai伺服器", "ai 伺服器", "伺服器", "server"))
            has_growth_2026 = self._evidence_contains(governance_report, ("2026", "倍增", "成長", "動能"))
            if has_ai_server and has_growth_2026 and share_value:
                return f"{label} 目前公開資訊顯示，AI 伺服器相關營收比重約 {share_value}%，且 2026 年成長仍被多個來源視為重要動能。"
            if has_ai_server and has_growth_2026:
                return f"{label} 目前公開資訊顯示，AI 伺服器仍被視為 2026 年的重要成長動能，但現有來源未一致揭露明確營收占比。"
            if has_ai_server:
                return f"{label} 目前已有 AI 伺服器相關資訊可供整理，但對 2026 年成長預測與營收占比仍缺乏足夠一致的公開證據。"
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}AI \u4f3a\u670d\u5668\u71df\u6536\u5360\u6bd4\u8207 2026 \u5e74\u6210\u9577\u9810\u6e2c\u3002"

        if query.question_type == "technical_indicator_review":
            rsi = self._extract_number(governance_report, r"RSI14 \u7d04 (\d+(?:\.\d+)?)")
            k_value = self._extract_number(governance_report, r"K \u503c\u7d04 (\d+(?:\.\d+)?)")
            d_value = self._extract_number(governance_report, r"D \u503c\u7d04 (\d+(?:\.\d+)?)")
            latest_close = self._extract_number(governance_report, r"\u6700\u65b0\u6536\u76e4\u50f9\u7d04 (\d+(?:\.\d+)?) \u5143")
            macd_line = self._extract_number(governance_report, r"MACD \u7dda\u7d04 (-?\d+(?:\.\d+)?)")
            signal_line = self._extract_number(governance_report, r"Signal \u7dda\u7d04 (-?\d+(?:\.\d+)?)")
            histogram = self._extract_number(governance_report, r"Histogram \u7d04 (-?\d+(?:\.\d+)?)")
            bollinger_position = self._extract_text(
                governance_report,
                r"(\u63a5\u8fd1\u5e03\u6797\u4e0a\u8ecc|\u63a5\u8fd1\u5e03\u6797\u4e0b\u8ecc|\u4f4d\u65bc\u5e03\u6797\u4e2d\u8ecc\u504f\u4e0a|\u4f4d\u65bc\u5e03\u6797\u4e2d\u8ecc\u504f\u4e0b|\u4f4d\u65bc\u4e2d\u8ecc\u9644\u8fd1)"
            )
            macd_trend = self._extract_text(governance_report, r"MACD \u52d5\u80fd([\u504f\u591a\u504f\u7a7a\u8f49\u5f37\u8f49\u5f31]+)")
            ma5_bias = self._extract_number(governance_report, r"MA5 \u4e56\u96e2\u7387\u7d04 (-?\d+(?:\.\d+)?)%")
            ma20_bias = self._extract_number(governance_report, r"MA20 \u4e56\u96e2\u7387\u7d04 (-?\d+(?:\.\d+)?)%")
            overbought = self._extract_text(governance_report, r"(\u5c1a\u672a\u9032\u5165\u8d85\u8cb7\u5340|\u75d1\u4f3c\u9032\u5165\u8d85\u8cb7\u5340|\u5df2\u9032\u5165\u8d85\u8cb7\u5340)")
            if rsi and k_value and d_value and overbought and macd_trend and bollinger_position and ma5_bias and ma20_bias:
                return (
                    f"{label} \u6700\u65b0\u6280\u8853\u6307\u6a19\u986f\u793a\uff0cRSI14 \u7d04 {rsi}\uff0c"
                    f"K \u503c\u7d04 {k_value}\uff0cD \u503c\u7d04 {d_value}\uff0c{overbought}\uff1b"
                    f"MACD \u52d5\u80fd{macd_trend}\uff0c\u80a1\u50f9{bollinger_position}\uff0c"
                    f"MA5 \u4e56\u96e2\u7387\u7d04 {ma5_bias}%\uff0cMA20 \u4e56\u96e2\u7387\u7d04 {ma20_bias}%\u3002"
                )
            if latest_close and macd_line and signal_line and histogram:
                return (
                    f"{label} \u76ee\u524d\u6700\u65b0\u6536\u76e4\u50f9\u7d04 {latest_close} \u5143\uff0c"
                    f"MACD \u7dda\u7d04 {macd_line}\uff0cSignal \u7dda\u7d04 {signal_line}\uff0cHistogram \u7d04 {histogram}\uff0c"
                    "\u4f46\u4ecd\u9700\u7d50\u5408 RSI\u3001KD \u8207\u5e03\u6797\u901a\u9053\u627f\u63a5\u8db3\u5920\u8b49\u64da\u5f8c\u624d\u80fd\u5b8c\u6574\u5224\u8b80\u3002"
                )
            if latest_close:
                return f"{label} \u76ee\u524d\u5df2\u6709\u6700\u65b0\u50f9\u683c\u8cc7\u6599\uff0c\u4f46\u4e0d\u8db3\u4ee5\u5b8c\u6574\u78ba\u8a8d RSI\u3001KD\u3001MACD\u3001\u5e03\u6797\u901a\u9053\u8207\u5747\u7dda\u4e56\u96e2\u7684\u7d9c\u5408\u5224\u8b80\u3002"
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u76ee\u524d\u7684 RSI\u3001KD\u3001MACD\u3001\u5e03\u6797\u901a\u9053\u8207\u5747\u7dda\u4e56\u96e2\u72c0\u614b\u3002"

        if query.question_type == "season_line_margin_review":
            latest_close = self._extract_number(governance_report, r"\u6700\u65b0\u6536\u76e4\u50f9\u7d04 (\d+(?:\.\d+)?) \u5143")
            season_line = self._extract_number(governance_report, r"\u5b63\u7dda\(MA60\)\u7d04 (\d+(?:\.\d+)?) \u5143")
            season_line_status = self._extract_text(governance_report, r"(\u8fd1\u671f\u8dcc\u7834\u5b63\u7dda|\u4ecd\u5728\u5b63\u7dda\u4e0b\u65b9|\u91cd\u65b0\u7ad9\u56de\u5b63\u7dda|\u5c1a\u672a\u8dcc\u7834\u5b63\u7dda)")
            margin_balance = self._extract_number(governance_report, r"\u6700\u65b0\u878d\u8cc7\u9918\u984d\u7d04 (\d+(?:\.\d+)?) \u5f35")
            utilization_pct = self._extract_number(governance_report, r"\u878d\u8cc7\u4f7f\u7528\u7387\u7d04 (\d+(?:\.\d+)?)%")
            average_delta_pct = self._extract_number(governance_report, r"\u76f8\u8f03\u8fd1 20 \u65e5\u5e73\u5747\u8b8a\u52d5\u7d04 (-?\d+(?:\.\d+)?)%")
            margin_status = self._extract_text(governance_report, r"\u7c4c\u78bc\u9762\u5c6c(\u504f\u9ad8|\u4e2d\u6027\u504f\u9ad8|\u4e2d\u6027\u504f\u4f4e|\u4e2d\u6027)")
            if latest_close and season_line and season_line_status and margin_balance and utilization_pct and margin_status:
                delta_segment = ""
                if average_delta_pct is not None:
                    delta_segment = f"\uff0c\u76f8\u8f03\u8fd1 20 \u65e5\u5e73\u5747\u8b8a\u52d5\u7d04 {average_delta_pct}%"
                return (
                    f"{label} \u6700\u65b0\u6536\u76e4\u50f9\u7d04 {latest_close} \u5143\uff0c\u5b63\u7dda\u7d04 {season_line} \u5143\uff0c"
                    f"\u76ee\u524d{season_line_status}\uff1b\u6700\u65b0\u878d\u8cc7\u9918\u984d\u7d04 {margin_balance} \u5f35\uff0c"
                    f"\u878d\u8cc7\u4f7f\u7528\u7387\u7d04 {utilization_pct}%{delta_segment}\uff0c\u82e5\u4ee5\u7c4c\u78bc\u9762\u63a8\u4f30\u5c6c{margin_status}\u3002"
                )
            if latest_close and season_line and season_line_status:
                return (
                    f"{label} \u6700\u65b0\u6536\u76e4\u50f9\u7d04 {latest_close} \u5143\uff0c\u5b63\u7dda\u7d04 {season_line} \u5143\uff0c"
                    f"\u76ee\u524d{season_line_status}\uff0c\u4f46\u878d\u8cc7\u9918\u984d\u8b49\u64da\u4ecd\u4e0d\u8db3\u4ee5\u5b8c\u6574\u8a55\u4f30\u5e02\u5834\u770b\u6cd5\u3002"
                )
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u80a1\u50f9\u662f\u5426\u8dcc\u7834\u5b63\u7dda\u53ca\u878d\u8cc7\u9918\u984d\u7684\u5e02\u5834\u89c0\u611f\u3002"

        if query.question_type == "shipping_rate_impact_review":
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
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d\u7d05\u6d77\u822a\u7dda\u53d7\u963b\u5c0d{shipping_label}\u7684 SCFI \u652f\u6490\u529b\u9053\u8207\u5206\u6790\u5e2b\u76ee\u6a19\u50f9\u8abf\u6574\u3002"

        if query.question_type == "electricity_cost_impact_review":
            comparison_label = query.comparison_company_name or query.comparison_ticker
            electricity_label = label if not comparison_label else f"{label}與{comparison_label}"
            has_tariff = self._evidence_contains(governance_report, ("工業電價", "電價", "調漲", "漲價", "電費"))
            has_cost = self._evidence_contains(governance_report, ("成本", "毛利", "壓力", "費用", "獲利"))
            has_amount = self._evidence_contains(governance_report, ("億元", "千萬元", "增加額度", "增幅", "%"))
            has_response = self._evidence_contains(governance_report, ("因應", "對策", "節能", "節電", "降耗", "轉嫁", "調價", "綠電", "自發電"))

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
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d\u5de5\u696d\u96fb\u50f9\u8abf\u6f32\u5c0d{electricity_label}\u7684\u6210\u672c\u589e\u52a0\u5e45\u5ea6\u8207\u56e0\u61c9\u5c0d\u7b56\u3002"

        if query.question_type == "macro_yield_sentiment_review":
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
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d\u7f8e\u570b CPI \u5c0d{label}\u8207\u9ad8\u6b96\u5229\u7387\u65cf\u7fa4\u7684\u8ca0\u9762\u60c5\u7dd2\u5f71\u97ff\u8207\u6cd5\u4eba\u89c0\u9ede\u3002"

        if query.question_type == "theme_impact_review":
            theme_label = label
            if query.comparison_company_name or query.comparison_ticker:
                comparison_label = query.comparison_company_name or query.comparison_ticker
                theme_label = f"{label}、{comparison_label}"

            has_cobalt_signal = self._evidence_contains(governance_report, ("\u9207\u50f9", "\u9207", "\u6cb9\u50f9", "\u539f\u6599", "cobalt"))
            has_energy_transition = self._evidence_contains(governance_report, ("\u80fd\u6e90\u8f49\u578b", "\u57ce\u5e02\u63a1\u7926"))
            has_product_mix = self._evidence_contains(governance_report, ("\u7522\u54c1\u7d44\u5408", "\u5eab\u5b58\u6548\u61c9", "\u7522\u80fd\u5229\u7528\u7387"))
            has_asml_signal = self._evidence_contains(governance_report, ("ASML", "\u827e\u53f8\u6469\u723e", "\u5c55\u671b", "\u4e0d\u5982\u9810\u671f", "\u4fdd\u5b88", "\u4e0b\u4fee"))
            has_equipment_signal = self._evidence_contains(governance_report, ("\u534a\u5c0e\u9ad4\u8a2d\u5099", "\u8a2d\u5099", "EUV", "\u66dd\u5149\u6a5f", "\u8cc7\u672c\u652f\u51fa", "\u64f4\u7522", "\u8a02\u55ae"))
            has_sentiment_signal = self._evidence_contains(governance_report, ("\u5229\u7a7a", "\u60c5\u7dd2", "\u89c0\u671b", "\u4fdd\u5b88", "\u964d\u6eab", "\u4e0b\u4fee"))
            if governance_report.evidence:
                points: list[str] = []
                if has_asml_signal:
                    points.append("ASML \u5c55\u671b\u504f\u4fdd\u5b88\u7684\u8a0a\u865f\uff0c\u5bb9\u6613\u88ab\u5e02\u5834\u89e3\u8b80\u70ba\u8a2d\u5099\u93c8\u77ed\u7dda\u9006\u98a8")
                if has_equipment_signal:
                    points.append("\u534a\u5c0e\u9ad4\u8a2d\u5099\u65cf\u7fa4\u7684\u89c0\u5bdf\u91cd\u9ede\uff0c\u6703\u56de\u5230\u6676\u5713\u5ee0\u8cc7\u672c\u652f\u51fa\u3001\u64f4\u7522\u7bc0\u594f\u8207\u8a02\u55ae\u80fd\u898b\u5ea6")
                if has_sentiment_signal:
                    points.append("\u77ed\u7dda\u60c5\u7dd2\u504f\u5411\u4fdd\u5b88\u6216\u89c0\u671b\uff0c\u5229\u7a7a\u89e3\u8b80\u901a\u5e38\u6703\u5148\u53cd\u6620\u5728\u984c\u6750\u71b1\u5ea6\u8207\u4f30\u503c")
                if has_energy_transition:
                    points.append("\u8fd1\u671f\u516c\u958b\u8cc7\u8a0a\u4ecd\u504f\u5411\u80fd\u6e90\u8f49\u578b\u8207\u6750\u6599\u984c\u6750")
                if has_product_mix:
                    points.append("\u77ed\u7dda\u8a0a\u865f\u4ecd\u4ee5\u7522\u54c1\u7d44\u5408\u3001\u5eab\u5b58\u6548\u61c9\u8207\u7522\u80fd\u5229\u7528\u7387\u70ba\u4e3b")
                if has_cobalt_signal:
                    points.append("\u539f\u6599\u50f9\u683c\u8b8a\u52d5\u4ecd\u662f\u5f71\u97ff\u6ce2\u52d5\u7684\u95dc\u9375")
                if points:
                    return f"{theme_label} \u76ee\u524d\u53ef\u6574\u7406\u5230\u7684\u6700\u65b0\u8a0a\u606f\u986f\u793a\uff0c" + "\uff1b".join(points) + "\u3002"
                return f"{theme_label} \u76ee\u524d\u6709\u90e8\u5206\u65b0\u805e\u53ef\u4f9b\u53c3\u8003\uff0c\u4f46\u516c\u958b\u8cc7\u8a0a\u4ecd\u4e0d\u8db3\u4ee5\u76f4\u63a5\u91cf\u5316 ASML \u5c55\u671b\u8b8a\u5316\u5c0d\u8a2d\u5099\u65cf\u7fa4\u57fa\u672c\u9762\u7684\u5be6\u969b\u50b3\u5c0e\u5f37\u5ea6\u3002"
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8dASML\u5c55\u671b\u8f49\u5f31\u5c0d{theme_label}\u7684\u6700\u65b0\u5229\u7a7a\u5206\u6790\u8207\u60c5\u7dd2\u5f71\u97ff\u3002"

        if query.question_type == "dividend_yield_review":
            cash_dividend = self._extract_number(governance_report, r"\u73fe\u91d1\u80a1\u5229(?:\u5408\u8a08)?\u7d04 (\d+(?:\.\d+)?) \u5143")
            close_price = self._extract_number(governance_report, r"\u6700\u65b0\u6536\u76e4\u50f9\u7d04 (\d+(?:\.\d+)?) \u5143")
            yield_pct = self._extract_number(governance_report, r"\u73fe\u91d1\u6b96\u5229\u7387\u7d04 (\d+(?:\.\d+)?)%")
            if cash_dividend and close_price and yield_pct:
                return f"{label} \u6700\u65b0\u53ef\u53d6\u5f97\u7684\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend} \u5143\uff1b\u82e5\u4ee5\u6700\u65b0\u6536\u76e4\u50f9\u7d04 {close_price} \u5143\u63db\u7b97\uff0c\u73fe\u91d1\u6b96\u5229\u7387\u7d04 {yield_pct}%\u3002"
            if cash_dividend:
                return f"{label} \u6700\u65b0\u53ef\u53d6\u5f97\u7684\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend} \u5143\uff0c\u4f46\u76ee\u524d\u4e0d\u8db3\u4ee5\u7a69\u5b9a\u63db\u7b97\u6b96\u5229\u7387\u3002"
            return f"{label} \u76ee\u524d\u5df2\u6709\u90e8\u5206\u80a1\u5229\u8cc7\u6599\uff0c\u4f46\u4e0d\u8db3\u4ee5\u5b8c\u6574\u78ba\u8a8d\u6700\u65b0\u914d\u606f\u653f\u7b56\u8207\u73fe\u91d1\u6b96\u5229\u7387\u3002"

        if query.question_type == "ex_dividend_performance":
            cash_dividend = self._extract_number(governance_report, r"\u73fe\u91d1\u80a1\u5229\u7d04 (\d+(?:\.\d+)?) \u5143")
            ex_date = self._extract_text(governance_report, r"\u9664\u606f\u4ea4\u6613\u65e5\u70ba (\d{4}-\d{2}-\d{2})")
            intraday_fill_ratio = self._extract_number(governance_report, r"\u76e4\u4e2d\u6700\u9ad8\u586b\u606f\u7387\u7d04 (\d+(?:\.\d+)?)%")
            close_fill_ratio = self._extract_number(governance_report, r"\u6536\u76e4\u586b\u606f\u7387\u7d04 (\d+(?:\.\d+)?)%")
            reaction = self._extract_text(governance_report, r"\u5e02\u5834\u53cd\u61c9\u504f([\u4e2d\u6027\u6b63\u5411\u5f37\u52c1\u5f31\u52e2]+)")
            same_day_fill = self._extract_text(governance_report, r"(\u7576\u5929(?:\u76e4\u4e2d)?(?:\u5b8c\u6210|\u672a\u5b8c\u6210)\u586b\u606f)")
            if ex_date and intraday_fill_ratio and close_fill_ratio:
                fill_sentence = f"\u9664\u606f\u65e5 {ex_date} \u76e4\u4e2d\u6700\u9ad8\u586b\u606f\u7387\u7d04 {intraday_fill_ratio}%\uff0c\u6536\u76e4\u586b\u606f\u7387\u7d04 {close_fill_ratio}%\u3002"
                if same_day_fill:
                    fill_sentence += same_day_fill + "\u3002"
                if reaction:
                    fill_sentence += f"\u5e02\u5834\u53cd\u61c9\u504f{reaction}\u3002"
                return f"{label} \u6700\u65b0\u4e00\u6b21\u9664\u606f\u4e8b\u4ef6\u986f\u793a\uff0c{fill_sentence}"
            if cash_dividend and ex_date:
                return f"{label} \u76ee\u524d\u53ef\u78ba\u8a8d\u6700\u65b0\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend} \u5143\uff0c\u9664\u606f\u65e5\u70ba {ex_date}\uff0c\u4f46\u4ecd\u7f3a\u5c11\u8db3\u5920\u50f9\u683c\u8b49\u64da\u4f86\u78ba\u8a8d\u7576\u5929\u7684\u586b\u606f\u8868\u73fe\u3002"
            return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}\u9664\u6b0a\u606f\u7576\u5929\u7684\u586b\u606f\u8868\u73fe\u8207\u5e02\u5834\u53cd\u61c9\u3002"

        if query.question_type == "eps_dividend_review":
            annual_eps = self._extract_number(governance_report, r"\u5168\u5e74 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
            latest_quarter_eps = self._extract_number(governance_report, r"\u6700\u65b0\u4e00\u5b63 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
            cash_dividend = self._extract_number(governance_report, r"\u73fe\u91d1\u80a1\u5229(?:\u5408\u8a08)?\u7d04 (\d+(?:\.\d+)?) \u5143")
            if annual_eps and cash_dividend:
                return f"{label} \u53bb\u5e74\u5168\u5e74 EPS \u7d04 {annual_eps} \u5143\uff1b\u76ee\u524d\u5df2\u53d6\u5f97\u7684\u80a1\u5229\u8cc7\u6599\u986f\u793a\uff0c\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend} \u5143\uff0c\u5e02\u5834\u9810\u671f\u4ecd\u9700\u914d\u5408\u6700\u65b0\u516c\u544a\u8207\u65b0\u805e\u89c0\u5bdf\u3002"
            if latest_quarter_eps and cash_dividend:
                return f"{label} \u6700\u65b0\u4e00\u5b63 EPS \u7d04 {latest_quarter_eps} \u5143\uff1b\u76ee\u524d\u53ef\u53d6\u5f97\u7684\u80a1\u5229\u8cc7\u6599\u986f\u793a\uff0c\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend} \u5143\u3002"
            if annual_eps:
                return f"{label} \u53bb\u5e74\u5168\u5e74 EPS \u7d04 {annual_eps} \u5143\uff0c\u4f46\u80a1\u5229\u9810\u671f\u4ecd\u9700\u7b49\u5f85\u66f4\u591a\u516c\u544a\u6216\u65b0\u805e\u4f50\u8b49\u3002"
            return f"{label} \u76ee\u524d\u5df2\u6709\u90e8\u5206\u8ca1\u5831\u6216\u80a1\u5229\u8cc7\u6599\uff0c\u4f46\u4ecd\u4e0d\u8db3\u4ee5\u5b8c\u6574\u78ba\u8a8d EPS \u8207\u5e02\u5834\u80a1\u5229\u9810\u671f\u3002"

        if query.question_type == "earnings_summary":
            annual_eps = self._extract_number(governance_report, r"\u5168\u5e74 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
            latest_quarter_eps = self._extract_number(governance_report, r"\u6700\u65b0\u4e00\u5b63 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
            if annual_eps:
                return f"{label} \u76ee\u524d\u53ef\u6574\u7406\u5230\u7684\u8ca1\u5831\u91cd\u9ede\u986f\u793a\uff0c\u53bb\u5e74\u5168\u5e74 EPS \u7d04 {annual_eps} \u5143\u3002"
            if latest_quarter_eps:
                return f"{label} \u76ee\u524d\u53ef\u6574\u7406\u5230\u7684\u6700\u65b0\u4e00\u5b63\u8ca1\u5831\u91cd\u9ede\u986f\u793a\uff0c EPS \u7d04 {latest_quarter_eps} \u5143\u3002"
            return f"{label} \u76ee\u524d\u6709\u8ca1\u5831\u76f8\u95dc\u8cc7\u6599\u53ef\u4f9b\u6574\u7406\uff0c\u4f46\u7d30\u7bc0\u4ecd\u61c9\u56de\u770b\u539f\u59cb\u63ed\u9732\u5167\u5bb9\u3002"

        if query.question_type == "announcement_summary":
            lead_title = governance_report.evidence[0].title if governance_report.evidence else "\u6700\u65b0\u516c\u544a"
            return f"{label} \u8fd1\u671f\u6709\u53ef\u8ffd\u8e64\u7684\u516c\u544a\u6216\u9ad8\u53ef\u4fe1\u8cc7\u6599\u66f4\u65b0\uff0c\u91cd\u9ede\u5305\u62ec\uff1a{lead_title}\u3002"

        official_count = sum(1 for item in governance_report.evidence if item.source_tier == SourceTier.HIGH)
        if official_count:
            return f"{label} \u76ee\u524d\u6709\u5b98\u65b9\u6216\u9ad8\u53ef\u4fe1\u4f86\u6e90\u53ef\u4f9b\u6574\u7406\uff0c\u4f46\u73fe\u6709\u8cc7\u8a0a\u4ecd\u61c9\u8207\u5f8c\u7e8c\u516c\u544a\u4e00\u4f75\u89c0\u5bdf\u3002"
        return f"{label} \u76ee\u524d\u6709\u90e8\u5206\u53ef\u53c3\u8003\u8cc7\u6599\uff0c\u4f46\u4f86\u6e90\u5f37\u5ea6\u8207\u4e00\u81f4\u6027\u4ecd\u9700\u6301\u7e8c\u78ba\u8a8d\u3002"

    def _build_impacts(self, query: StructuredQuery) -> list[str]:
        if query.question_type == "guidance_reaction_review":
            return [
                "法說後的媒體與法人反應，可協助觀察市場如何解讀公司對下半年需求、毛利與出貨節奏的說法。",
                "若正面與負面反應同時存在，通常代表市場對指引的可持續性仍有分歧。",
                "比起單一標題，交叉比對多家媒體與法人觀點，更能看出真正的焦點與疑慮。",
            ]
        if query.question_type == "listing_revenue_review":
            return [
                "轉上市後的股價波動常同時受籌碼、題材與營運消息影響，不能只用單一新聞解釋。",
                "月營收若明顯年增，通常代表需求或航線布局有支撐，但不等於獲利會同步放大。",
                "若市場消息主要集中在品牌、票價或新航線題材，股價反應也可能比基本面更快。",
            ]
        if query.question_type == "price_range":
            return [
                "\u5340\u9593\u9ad8\u4f4e\u9ede\u53ef\u7528\u4f86\u89c0\u5bdf\u77ed\u7dda\u6ce2\u52d5\u8207\u5e02\u5834\u4ea4\u6613\u71b1\u5ea6\u3002",
                "\u82e5\u50f9\u683c\u9760\u8fd1\u5340\u9593\u4e0a\u7de3\u6216\u4e0b\u7de3\uff0c\u9700\u642d\u914d\u5f8c\u7e8c\u6d88\u606f\u78ba\u8a8d\u662f\u5426\u5ef6\u7e8c\u3002",
                "\u55ae\u770b\u5340\u9593\u7121\u6cd5\u5224\u65b7\u57fa\u672c\u9762\u662f\u5426\u540c\u6b65\u6539\u5584\u6216\u8f49\u5f31\u3002",
            ]
        if query.question_type == "profitability_stability_review":
            return [
                "\u62ff\u8fd1\u4e94\u5e74\u7684\u5e74\u5ea6\u7372\u5229\u4e00\u8d77\u770b\uff0c\u6bd4\u55ae\u770b\u6700\u65b0\u4e00\u5e74\u66f4\u80fd\u5224\u65b7\u9019\u6a94\u80a1\u7968\u662f\u5426\u9069\u5408\u7576\u6210\u9577\u671f\u73fe\u91d1\u6d41\u578b\u90e8\u4f4d\u89c0\u5bdf\u3002",
                "\u82e5\u8fd1\u5e74\u51fa\u73fe\u8f49\u8667\uff0c\u5c31\u4e0d\u9069\u5408\u76f4\u63a5\u8996\u70ba\u300c\u7a69\u5b9a\u7372\u5229\u300d\u516c\u53f8\uff0c\u5f8c\u7e8c\u8981\u518d\u770b\u662f\u9031\u671f\u6027\u9084\u662f\u7d50\u69cb\u6027\u554f\u984c\u3002",
                "\u82e5\u8655\u65bc\u8f49\u8667\u5e74\uff0c\u80fd\u518d\u62c6\u51fa\u662f\u672c\u696d\u8f49\u5f31\u9084\u662f\u71df\u696d\u5916\u62d6\u7d2f\uff0c\u6703\u6bd4\u53ea\u770b EPS \u66f4\u6709\u53c3\u8003\u50f9\u503c\u3002",
            ]
        if query.question_type == "gross_margin_comparison_review":
            return [
                "\u6bdb\u5229\u7387\u8f03\u9ad8\u7684\u4e00\u65b9\uff0c\u901a\u5e38\u4ee3\u8868\u5b9a\u50f9\u80fd\u529b\u3001\u8ca8\u6e90\u7d50\u69cb\u6216\u6210\u672c\u63a7\u7ba1\u76f8\u5c0d\u8f03\u4f54\u512a\u52e2\u3002",
                "\u82e5\u5169\u5bb6\u516c\u53f8\u7684\u6bdb\u5229\u7387\u5dee\u8ddd\u80fd\u5728\u8fd1\u5e7e\u5b63\u6301\u7e8c\uff0c\u8f03\u5bb9\u6613\u53cd\u6620\u70ba\u7d50\u69cb\u6027\u5dee\u7570\uff0c\u4e0d\u53ea\u662f\u55ae\u5b63\u6ce2\u52d5\u3002",
                "\u4e0d\u904e\u300c\u7d93\u71df\u6548\u7387\u300d\u4e0d\u80fd\u53ea\u770b\u6bdb\u5229\u7387\uff0c\u4ecd\u8981\u642d\u914d\u71df\u76ca\u7387\u3001\u8cbb\u7528\u7387\u548c\u8cc7\u7522\u9031\u8f49\u4e00\u8d77\u89c0\u5bdf\u3002",
            ]
        if query.question_type == "margin_turnaround_review":
            return [
                "毛利率是否由負轉正，能快速判斷產品組合、報價與成本吸收能力是否有回到健康區間。",
                "營業利益是否同步轉正，比單看毛利率更能判斷本業是否真的擺脫虧損。",
                "若毛利率改善但營業利益仍為負，通常代表費用、折舊或稼動率壓力仍在，本業修復尚未完成。",
            ]
        if query.question_type == "monthly_revenue_yoy_review":
            return [
                "\u6708\u71df\u6536\u7d2f\u8a08\u5e74\u589e\u7387\u53ef\u4ee5\u5feb\u901f\u89c0\u5bdf\u516c\u53f8\u7576\u5e74\u958b\u5c40\u7684\u71df\u904b\u52d5\u80fd\u3002",
                "\u62ff\u4eca\u5e74\u524d\u5e7e\u500b\u6708\u8207\u53bb\u5e74\u540c\u671f\u6bd4\u8f03\uff0c\u80fd\u907f\u514d\u55ae\u770b\u55ae\u6708\u6578\u5b57\u53d7\u57fa\u671f\u6216\u51fa\u8ca8\u6642\u9ede\u5f71\u97ff\u3002",
                "\u71df\u6536\u5e74\u589e\u53ef\u4ee5\u7576\u4f5c\u57fa\u672c\u9762\u71b1\u5ea6\u7684\u5148\u884c\u6307\u6a19\uff0c\u4f46\u4ecd\u9700\u642d\u914d\u6bdb\u5229\u7387\u8207 EPS \u624d\u80fd\u5b8c\u6574\u89e3\u8b80\u3002",
            ]
        if query.question_type == "debt_dividend_safety_review":
            return [
                "負債比率可以協助觀察公司近幾期的槓桿水位是否突然墊高。",
                "把最新負債比率和前一季、去年同期一起比較，較能分辨是短期波動還是財務結構轉弱。",
                "現金及約當現金若明顯高於現金股利總額，通常代表眼前的股利支付緩衝仍在。",
            ]
        if query.question_type == "fcf_dividend_sustainability_review":
            return [
                "\u81ea\u7531\u73fe\u91d1\u6d41\u53ef\u4ee5\u5e6b\u52a9\u89c0\u5bdf\u516c\u53f8\u5728\u7dad\u6301\u8cc7\u672c\u652f\u51fa\u4e4b\u5f8c\uff0c\u9084\u5269\u4e0b\u591a\u5c11\u73fe\u91d1\u80fd\u652f\u61c9\u80a1\u5229\u6216\u5176\u4ed6\u8cc7\u91d1\u7528\u9014\u3002",
                "\u628a\u73fe\u91d1\u80a1\u5229\u767c\u653e\u7e3d\u984d\u8207 FCF \u4e00\u8d77\u770b\uff0c\u6bd4\u55ae\u770b\u6b96\u5229\u7387\u66f4\u80fd\u5224\u65b7\u80a1\u5229\u653f\u7b56\u662f\u5426\u6709\u73fe\u91d1\u6d41\u652f\u6490\u3002",
                "\u82e5\u9023\u7e8c\u591a\u5e74 FCF \u9ad8\u65bc\u73fe\u91d1\u80a1\u5229\u652f\u51fa\uff0c\u901a\u5e38\u4ee3\u8868\u73fe\u968e\u6bb5\u80a1\u5229\u653f\u7b56\u8f03\u5177\u7a69\u5b9a\u57fa\u790e\u3002",
            ]
        if query.question_type == "pe_valuation_review":
            return [
                "\u672c\u76ca\u6bd4\u80fd\u5e6b\u52a9\u4f30\u8a08\u5e02\u5834\u76ee\u524d\u9858\u610f\u7528\u591a\u9ad8\u7684\u8a55\u50f9\u500d\u6578\u53cd\u6620\u516c\u53f8\u7372\u5229\u80fd\u529b\u3002",
                "\u628a\u76ee\u524d\u672c\u76ca\u6bd4\u653e\u56de\u8fd1 13 \u500b\u6708\u6b77\u53f2\u5340\u9593\uff0c\u53ef\u4ee5\u6bd4\u55ae\u770b\u55ae\u4e00\u500d\u6578\u66f4\u5bb9\u6613\u5224\u65b7\u4f30\u503c\u662f\u5426\u504f\u9ad8\u6216\u504f\u4f4e\u3002",
                "\u5c0d\u9577\u671f\u6295\u8cc7\u4f86\u8aaa\uff0c\u672c\u76ca\u6bd4\u53ea\u662f\u9032\u5834\u4ef7\u683c\u7684\u4e00\u500b\u89c0\u5bdf\u9762\uff0c\u4ecd\u9700\u642d\u914d\u6210\u9577\u6027\u8207\u73fe\u91d1\u6d41\u4e00\u8d77\u770b\u3002",
            ]
        if query.question_type == "technical_indicator_review":
            return [
                "RSI \u8207 KD \u80fd\u5e6b\u52a9\u89c0\u5bdf\u77ed\u671f\u8cb7\u76e4\u71b1\u5ea6\u8207\u5f37\u5f31\u8b8a\u5316\u3002",
                "MACD \u9069\u5408\u7528\u4f86\u770b\u50f9\u683c\u52d5\u80fd\u662f\u5426\u6301\u7e8c\u8f49\u5f37\u6216\u8f49\u5f31\uff0c\u5e03\u6797\u901a\u9053\u5247\u53ef\u4ee5\u89c0\u5bdf\u80a1\u50f9\u76f8\u5c0d\u5340\u9593\u4f4d\u7f6e\u3002",
                "\u5747\u7dda\u4e56\u96e2\u53ef\u7528\u4f86\u6aa2\u67e5\u77ed\u7dda\u6f32\u52e2\u662f\u5426\u904e\u71b1\u6216\u904e\u51b7\uff0c\u642d\u914d\u591a\u500b\u6307\u6a19\u6703\u6bd4\u55ae\u770b RSI \u6216 KD \u66f4\u5b8c\u6574\u3002",
            ]
        if query.question_type == "shipping_rate_impact_review":
            return [
                "紅海航線受阻與 SCFI 變化，常是貨櫃航運股短線評價最直接的事件指標。",
                "把運價支撐與分析師目標價一起看，比單看題材新聞更能分辨市場是在交易基本面，還是在交易情緒。",
                "若長榮與陽明都被同一波航運新聞反覆提及，通常代表市場正在把它們視為同一組運價受惠標的。",
            ]
        if query.question_type == "electricity_cost_impact_review":
            return [
                "工業電價調漲通常會先影響高耗電產業的成本結構，再慢慢反映到毛利率與報價策略。",
                "把成本壓力和公司因應對策一起看，比只看漲價新聞更能判斷衝擊是短期還是可被吸收。",
                "若中鋼、台泥這類用電大戶都被同一波電價新聞反覆點名，通常代表市場正在重新評估其成本轉嫁能力。",
            ]
        if query.question_type == "macro_yield_sentiment_review":
            return [
                "美國 CPI 與通膨預期，常會先透過利率與債息路徑影響高殖利率股的評價。",
                "把高殖利率個股與金控股放在一起看，可以更快分辨市場是在交易殖利率吸引力，還是在交易利率風險。",
                "法人觀點若同時提到防禦性與壓力，通常代表市場並非全面看空，而是在重新校正估值。",
            ]
        if query.question_type == "theme_impact_review":
            return [
                "\u4e3b\u984c\u578b\u554f\u984c\u8981\u5148\u78ba\u8a8d\u6709\u6c92\u6709\u516c\u958b\u8cc7\u8a0a\u76f4\u63a5\u63d0\u5230\u8a72\u50b3\u5c0e\u95dc\u4fc2\uff0c\u4e0d\u9069\u5408\u53ea\u9760\u7522\u696d\u60f3\u50cf\u88dc\u7a7a\u3002",
                "\u5c0d\u534a\u5c0e\u9ad4\u8a2d\u5099\u65cf\u7fa4\u4f86\u8aaa\uff0c\u77ed\u7dda\u66f4\u5e38\u898b\u7684\u89c0\u5bdf\u9ede\u6703\u662f\u6676\u5713\u5ee0\u8cc7\u672c\u652f\u51fa\u3001\u64f4\u7522\u7bc0\u594f\u3001\u8a02\u55ae\u80fd\u898b\u5ea6\u8207\u4f30\u503c\u60c5\u7dd2\u3002",
                "\u82e5\u6c92\u6709\u66f4\u76f4\u63a5\u7684\u8a02\u55ae\u3001\u51fa\u8ca8\u6216\u516c\u53f8\u8aaa\u660e\uff0c\u5c0d\u300c\u5c55\u671b\u4e0d\u5982\u9810\u671f\u300d\u7684\u50b3\u5c0e\u5e45\u5ea6\u61c9\u4fdd\u6301\u4f4e\u78ba\u4fe1\u5ea6\u3002",
            ]
        if query.question_type == "revenue_growth_review":
            return [
                "AI 伺服器營收占比屬於基本面結構資訊，應優先看公司說法、法說與高可信新聞整理。",
                "2026 年成長預測通常來自管理層展望、產能擴張與產業需求判讀，不應直接當成已實現結果。",
                "若只有成長敘事而沒有一致數字，較適合解讀為方向性訊號，而不是精確預估。",
            ]
        if query.question_type == "season_line_margin_review":
            return [
                "\u5b63\u7dda\u662f\u4e2d\u77ed\u671f\u8da8\u52e2\u7684\u5e38\u898b\u89c0\u5bdf\u7dda\uff0c\u80fd\u5feb\u901f\u5efa\u7acb\u76ee\u524d\u50f9\u683c\u5f37\u5f31\u7684\u80cc\u666f\u3002",
                "\u878d\u8cc7\u9918\u984d\u8207\u878d\u8cc7\u4f7f\u7528\u7387\u80fd\u5354\u52a9\u89c0\u5bdf\u69d3\u687f\u8cc7\u91d1\u662f\u5426\u96c6\u4e2d\uff0c\u5c0d\u77ed\u7dda\u6ce2\u52d5\u901a\u5e38\u6bd4\u8f03\u654f\u611f\u3002",
                "\u628a\u5b63\u7dda\u4f4d\u7f6e\u8207\u878d\u8cc7\u9918\u984d\u4e00\u8d77\u770b\uff0c\u6bd4\u55ae\u770b\u50f9\u683c\u6216\u55ae\u770b\u7c4c\u78bc\u66f4\u5bb9\u6613\u638c\u63e1\u98a8\u96aa\u3002",
            ]
        if query.question_type == "dividend_yield_review":
            return [
                "\u80a1\u5229\u653f\u7b56\u53ef\u53cd\u6620\u516c\u53f8\u5c0d\u73fe\u91d1\u56de\u994b\u7684\u5b89\u6392\u3002",
                "\u73fe\u91d1\u6b96\u5229\u7387\u53ef\u5354\u52a9\u4f30\u7b97\u4ee5\u76ee\u524d\u80a1\u50f9\u63db\u7b97\u7684\u73fe\u91d1\u56de\u994b\u6c34\u6e96\u3002",
                "\u6b96\u5229\u7387\u4ecd\u6703\u96a8\u80a1\u50f9\u8b8a\u52d5\uff0c\u4e0d\u662f\u56fa\u5b9a\u503c\u3002",
            ]
        if query.question_type == "ex_dividend_performance":
            return [
                "\u9664\u6b0a\u606f\u7576\u5929\u7684\u586b\u606f\u7387\u53ef\u4ee5\u53cd\u6620\u5e02\u5834\u5c0d\u80a1\u5229\u984c\u6750\u7684\u63a5\u53d7\u5ea6\u3002",
                "\u76e4\u4e2d\u6700\u9ad8\u586b\u606f\u7387\u8207\u6536\u76e4\u586b\u606f\u7387\u53ef\u5354\u52a9\u5340\u5206\u662f\u77ed\u7dda\u8861\u52d5\u9084\u662f\u8cb7\u76e4\u5ef6\u7e8c\u3002",
                "\u4ea4\u6613\u91cf\u8207\u7576\u65e5\u50f9\u683c\u53cd\u61c9\u4e00\u8d77\u770b\uff0c\u6bd4\u55ae\u770b\u6f32\u8dcc\u66f4\u80fd\u63cf\u8ff0\u5e02\u5834\u614b\u5ea6\u3002",
            ]
        if query.question_type in {"earnings_summary", "eps_dividend_review"}:
            return [
                "EPS \u8868\u73fe\u53ef\u53cd\u6620\u516c\u53f8\u7372\u5229\u80fd\u529b\u8207\u8cc7\u672c\u652f\u51fa\u627f\u53d7\u5ea6\u3002",
                "\u80a1\u5229\u8cc7\u8a0a\u5e38\u5f71\u97ff\u5e02\u5834\u5c0d\u73fe\u91d1\u56de\u994b\u8207\u6b96\u5229\u7387\u7684\u9810\u671f\u3002",
                "\u82e5\u8ca1\u5831\u8207\u80a1\u5229\u65b9\u5411\u4e00\u81f4\uff0c\u901a\u5e38\u66f4\u5bb9\u6613\u5f62\u6210\u7a69\u5b9a\u6558\u4e8b\u3002",
            ]
        if query.question_type == "announcement_summary":
            return [
                "\u516c\u544a\u901a\u5e38\u6703\u5f71\u97ff\u5e02\u5834\u5c0d\u516c\u53f8\u77ed\u671f\u4e8b\u4ef6\u8207\u6c7a\u7b56\u7bc0\u594f\u7684\u7406\u89e3\u3002",
                "\u82e5\u516c\u544a\u6d89\u53ca\u80a1\u5229\u3001\u8463\u4e8b\u6703\u6216\u6cd5\u8aaa\uff0c\u53ef\u80fd\u63d0\u9ad8\u5f8c\u7e8c\u95dc\u6ce8\u5ea6\u3002",
                "\u516c\u544a\u672c\u8eab\u4ecd\u9700\u642d\u914d\u8ca1\u5831\u8207\u65b0\u805e\u4ea4\u53c9\u9a57\u8b49\uff0c\u907f\u514d\u904e\u5ea6\u89e3\u8b80\u3002",
            ]
        if is_fundamental_valuation_question(query):
            return [
                "\u57fa\u672c\u9762\u53ef\u4ee5\u5e6b\u52a9\u5224\u65b7\u7372\u5229\u8207\u71df\u904b\u52d5\u80fd\u662f\u5426\u5177\u5099\u5ef6\u7e8c\u6027\u3002",
                "\u672c\u76ca\u6bd4\u5247\u53ef\u4ee5\u62ff\u4f86\u5b9a\u4f4d\u5e02\u5834\u76ee\u524d\u7d66\u9019\u6a94\u516c\u53f8\u7684\u4f30\u503c\u6c34\u4f4d\u3002",
                "\u82e5\u8981\u505a\u9032\u5834\u5224\u65b7\uff0c\u6700\u597d\u662f\u628a\u7372\u5229\u8da8\u52e2\u548c\u4f30\u503c\u4f4d\u7f6e\u4e00\u8d77\u770b\u3002",
            ]
        if query.question_type == "price_outlook":
            return [
                "\u77ed\u671f\u6f32\u8dcc\u901a\u5e38\u540c\u6642\u53d7\u5230\u57fa\u672c\u9762\u3001\u8cc7\u91d1\u9762\u8207\u6d88\u606f\u9762\u5f71\u97ff\u3002",
                "\u82e5\u8fd1\u671f\u6709\u516c\u544a\u6216\u8ca1\u5831\u66f4\u65b0\uff0c\u5e02\u5834\u53cd\u61c9\u53ef\u80fd\u653e\u5927\u6ce2\u52d5\u3002",
                "\u7f3a\u5c11\u591a\u4f86\u6e90\u4f50\u8b49\u6642\uff0c\u4e0d\u5b9c\u628a\u55ae\u4e00\u8a0a\u865f\u89e3\u8b80\u6210\u660e\u78ba\u65b9\u5411\u3002",
            ]
        if query.topic == Topic.NEWS:
            return [
                "\u65b0\u805e\u80fd\u5e6b\u52a9\u7406\u89e3\u5e02\u5834\u76ee\u524d\u95dc\u6ce8\u7684\u4e8b\u4ef6\u8207\u6558\u4e8b\u3002",
                "\u82e5\u591a\u500b\u4f86\u6e90\u96c6\u4e2d\u5831\u5c0e\u540c\u4e00\u4e3b\u984c\uff0c\u4ee3\u8868\u95dc\u6ce8\u5ea6\u53ef\u80fd\u5347\u9ad8\u3002",
                "\u4ecd\u9700\u56de\u770b\u539f\u6587\u8207\u516c\u544a\uff0c\u907f\u514d\u53ea\u770b\u6458\u8981\u9020\u6210\u8aa4\u8b80\u3002",
            ]
        return [
            "\u73fe\u6709\u8cc7\u6599\u53ef\u4f5c\u70ba\u5feb\u901f\u638c\u63e1\u8108\u7d61\u7684\u8d77\u9ede\u3002",
            "\u82e5\u4e0d\u540c\u4f86\u6e90\u80fd\u4e92\u76f8\u5370\u8b49\uff0c\u5224\u8b80\u7a69\u5b9a\u5ea6\u6703\u66f4\u9ad8\u3002",
            "\u5f8c\u7e8c\u4ecd\u61c9\u6301\u7e8c\u8ffd\u8e64\u65b0\u7684\u516c\u544a\u3001\u8ca1\u5831\u8207\u65b0\u805e\u8b8a\u5316\u3002",
        ]

    def _build_risks(self, query: StructuredQuery, governance_report: GovernanceReport) -> list[str]:
        if query.question_type == "guidance_reaction_review":
            risks = [
                "法說後的媒體與法人解讀常帶有主觀判斷，未必等同公司最終實際營運結果。",
                "若只有少數報導提到下半年指引，市場情緒可能被單一敘事放大。",
                "即使會後反應偏正面，仍需追蹤後續月營收、財測與客戶拉貨節奏是否驗證。",
            ]
        elif query.question_type == "listing_revenue_review":
            risks = [
                "轉上市初期的價格波動可能夾雜籌碼與情緒因素，未必完全反映中長期基本面。",
                "單月營收年增或月增偏強，仍可能受旺季、航線擴張或一次性因素影響。",
                "若缺少公司法說或更直接公告佐證，對股價波動主因的解讀仍應保守。",
            ]
        elif query.question_type == "price_range":
            risks = [
                "\u50f9\u683c\u5340\u9593\u53ea\u53cd\u6620\u6b77\u53f2\u4ea4\u6613\u7d50\u679c\uff0c\u4e0d\u80fd\u76f4\u63a5\u63a8\u8ad6\u672a\u4f86\u65b9\u5411\u3002",
                "\u82e5\u671f\u9593\u5167\u525b\u597d\u51fa\u73fe\u91cd\u5927\u4e8b\u4ef6\uff0c\u5340\u9593\u53ef\u80fd\u5931\u771f\u653e\u5927\u77ed\u671f\u6ce2\u52d5\u3002",
                "\u672a\u7d50\u5408\u57fa\u672c\u9762\u8207\u516c\u544a\u6642\uff0c\u5bb9\u6613\u5ffd\u7565\u8da8\u52e2\u53cd\u8f49\u98a8\u96aa\u3002",
            ]
        elif query.question_type == "profitability_stability_review":
            risks = [
                "\u5c31\u7b97\u8fd1\u5e7e\u5e74\u90fd\u6709\u7372\u5229\uff0c\u82e5\u7372\u5229\u5e45\u5ea6\u8d77\u4f0f\u5f88\u5927\uff0c\u5c0d\u300c\u9000\u4f11\u5b58\u80a1\u300d\u7684\u7a69\u5b9a\u6027\u4ecd\u8981\u4fdd\u5b88\u770b\u5f85\u3002",
                "\u8f49\u8667\u5e74\u7684\u539f\u56e0\u82e5\u53ea\u80fd\u5f9e\u8ca1\u5831\u7d50\u69cb\u63a8\u4f30\uff0c\u4ecd\u61c9\u8207\u516c\u53f8\u6cd5\u8aaa\u6216\u5e74\u5831\u8aaa\u660e\u4e92\u76f8\u5c0d\u7167\u3002",
                "\u50b3\u7d71\u7522\u696d\u7372\u5229\u5bb9\u6613\u53d7\u666f\u6c23\u3001\u539f\u6599\u5831\u50f9\u548c\u5229\u5dee\u5f71\u97ff\uff0c\u904e\u53bb\u4e94\u5e74\u4e0d\u4e00\u5b9a\u80fd\u76f4\u63a5\u4ee3\u8868\u672a\u4f86\u4e94\u5e74\u3002",
            ]
        elif query.question_type == "gross_margin_comparison_review":
            risks = [
                "\u6bdb\u5229\u7387\u8f03\u9ad8\u4e0d\u4e00\u5b9a\u4ee3\u8868\u6574\u9ad4\u7d93\u71df\u6548\u7387\u5c31\u66f4\u597d\uff0c\u4ecd\u8981\u770b\u71df\u76ca\u7387\u3001\u8cbb\u7528\u63a7\u7ba1\u8207\u73fe\u91d1\u6d41\u3002",
                "\u822a\u904b\u80a1\u6bdb\u5229\u7387\u5bb9\u6613\u53d7\u904b\u50f9\u5468\u671f\u3001\u71c3\u6cb9\u6210\u672c\u8207\u822a\u7dda\u7d50\u69cb\u5f71\u97ff\uff0c\u55ae\u4e00\u5b63\u6578\u5b57\u53ef\u80fd\u6ce2\u52d5\u5f88\u5927\u3002",
                "\u82e5\u6bd4\u8f03\u6642\u9ede\u4e0d\u5b8c\u5168\u4e00\u81f4\uff0c\u6216\u8ca1\u5831\u53e3\u5f91\u525b\u597d\u53d7\u4e00\u6b21\u6027\u56e0\u7d20\u5f71\u97ff\uff0c\u89e3\u8b80\u4e0a\u8981\u4fdd\u5b88\u3002",
            ]
        elif query.question_type == "margin_turnaround_review":
            risks = [
                "毛利率轉正不一定等於整體獲利體質已經穩定改善，仍要看營業利益是否同步回正。",
                "若營業利益仍為負，代表費用結構、折舊負擔或稼動率壓力可能還在，本業復甦未必已站穩。",
                "單一季度的轉正也可能受匯率、產品組合或一次性因素影響，還需要連續幾季追蹤。",
            ]
        elif query.question_type == "monthly_revenue_yoy_review":
            risks = [
                "\u7d2f\u8a08\u71df\u6536\u5e74\u589e\u53ea\u53cd\u6620\u71df\u6536\u7aef\u7684\u8b8a\u5316\uff0c\u4e0d\u76f4\u63a5\u4ee3\u8868\u7345\u5229\u7387\u6216\u7372\u5229\u540c\u6b65\u6539\u5584\u3002",
                "\u55ae\u770b\u524d\u5e7e\u500b\u6708\u7684\u7d2f\u8a08\u71df\u6536\uff0c\u4ecd\u53ef\u80fd\u53d7\u51fa\u8ca8\u6642\u9ede\u3001\u532f\u7387\u6216\u5b63\u7bc0\u6027\u56e0\u7d20\u5f71\u97ff\u3002",
                "\u82e5\u6c92\u6709\u540c\u6642\u642d\u914d\u8ca1\u5831\u8207\u6cd5\u8aaa\uff0c\u5bb9\u6613\u904e\u5ea6\u628a\u71df\u6536\u6210\u9577\u89e3\u8b80\u6210\u7372\u5229\u78ba\u5b9a\u6027\u3002",
            ]
        elif query.question_type == "debt_dividend_safety_review":
            risks = [
                "負債比率短期變動不一定代表財務體質立即惡化，仍需拆解是應付帳款、借款或營運週轉造成。",
                "帳上現金高於股利總額，只能說目前支付緩衝存在，仍需搭配未來現金流與資本支出一起看。",
                "若後續公司調整股利政策、擴大投資或出現一次性資金需求，現有支應能力判斷也可能改變。",
            ]
        elif query.question_type == "fcf_dividend_sustainability_review":
            risks = [
                "\u81ea\u7531\u73fe\u91d1\u6d41\u96d6\u80fd\u8aaa\u660e\u73fe\u91d1\u652f\u61c9\u80fd\u529b\uff0c\u4ecd\u4e0d\u4ee3\u8868\u672a\u4f86\u80a1\u5229\u4e00\u5b9a\u7dad\u6301\u4e0d\u8b8a\u3002",
                "\u82e5\u5f8c\u7e8c\u8cc7\u672c\u652f\u51fa\u63d0\u5347\u3001\u73fe\u91d1\u6d41\u8f49\u5f31\u6216\u76e3\u7406\u653f\u7b56\u6539\u8b8a\uff0c\u80a1\u5229\u653f\u7b56\u4ecd\u53ef\u80fd\u8abf\u6574\u3002",
                "\u73fe\u91d1\u80a1\u5229\u767c\u653e\u7e3d\u984d\u70ba\u63a8\u4f30\u503c\uff0c\u4ecd\u61c9\u4ee5\u516c\u53f8\u6b63\u5f0f\u516c\u544a\u8207\u5be6\u969b\u767c\u653e\u5b89\u6392\u70ba\u6e96\u3002",
            ]
        elif query.question_type == "pe_valuation_review":
            risks = [
                "\u672c\u76ca\u6bd4\u6703\u96a8\u80a1\u50f9\u8207\u7372\u5229\u9810\u671f\u8b8a\u52d5\uff0c\u4eca\u5929\u7684\u4f30\u503c\u4f4d\u7f6e\u4e0d\u4ee3\u8868\u4e4b\u5f8c\u4e0d\u6703\u7e7c\u7e8c\u4e0a\u8abf\u6216\u4e0b\u4fee\u3002",
                "\u6b77\u53f2\u504f\u9ad8\u4e0d\u4e00\u5b9a\u4ee3\u8868\u9a6c\u4e0a\u8cb4\u5230\u4e0d\u80fd\u8cb7\uff0c\u4ecd\u9700\u642d\u914d\u672a\u4f86 EPS \u6210\u9577\u8207\u7522\u696d\u666f\u6c23\u5224\u65b7\u3002",
                "\u82e5\u53ea\u7528\u672c\u76ca\u6bd4\u6c7a\u5b9a\u662f\u5426\u9032\u5834\uff0c\u53ef\u80fd\u5ffd\u7565\u73fe\u91d1\u6d41\u3001\u80a1\u5229\u8207\u8cc7\u672c\u652f\u51fa\u7b49\u66f4\u9069\u5408\u9577\u6295\u7684\u8b8a\u6578\u3002",
            ]
        elif query.question_type == "technical_indicator_review":
            risks = [
                "\u8d85\u8cb7\u4e0d\u4ee3\u8868\u80a1\u50f9\u6703\u7acb\u5373\u56de\u6a94\uff0c\u4ecd\u53ef\u80fd\u56e0\u5f37\u52c1\u8da8\u52e2\u6301\u7e8c\u4e0a\u884c\u3002",
                "\u6280\u8853\u6307\u6a19\u662f\u6839\u64da\u6b77\u53f2\u50f9\u683c\u63a8\u7b97\uff0c\u9047\u5230\u7a81\u767c\u6d88\u606f\u6642\u53ef\u80fd\u5f88\u5feb\u5931\u771f\u3002",
                "\u82e5 MACD \u8207 RSI\u3001KD \u8a0a\u865f\u4e0d\u4e00\u81f4\uff0c\u8868\u793a\u77ed\u7dda\u52d5\u80fd\u8207\u50f9\u683c\u4f4d\u968e\u53ef\u80fd\u6b63\u5728\u62c9\u6240\uff0c\u4e0d\u5b9c\u55ae\u770b\u55ae\u4e00\u6307\u6a19\u4e0b\u7d50\u8ad6\u3002",
            ]
        elif query.question_type == "shipping_rate_impact_review":
            risks = [
                "SCFI 與紅海事件多屬短線催化，未必能直接等同全年獲利或長期運價中樞上移。",
                "分析師目標價調整常受事件時點與市場情緒影響，方向可能很快反轉。",
                "若後續紅海繞道、塞港或運力供給狀況緩解，短線支撐力道也可能同步降溫。",
            ]
        elif query.question_type == "electricity_cost_impact_review":
            risks = [
                "電價調漲帶來的成本壓力不一定能完整轉嫁，對毛利率的影響仍要看產品報價與景氣狀況。",
                "若公開資訊沒有揭露可精算的用電成本結構，單一公司增加額度多半只能做方向性判讀。",
                "公司宣示的節能或因應方案，未必能在短期內完全抵消漲價衝擊。",
            ]
        elif query.question_type == "macro_yield_sentiment_review":
            risks = [
                "CPI 對高殖利率股的影響常先反映在評價與情緒面，不一定立刻反映到基本面。",
                "若市場很快把焦點轉回降息時點或債息回落，高殖利率股的負面情緒也可能快速修正。",
                "法人觀點常同時包含防禦性與估值壓力，若只看單一方向容易過度解讀。",
            ]
        elif query.question_type == "theme_impact_review":
            risks = [
                "\u4e3b\u984c\u8207\u500b\u80a1\u95dc\u806f\u4e0d\u4ee3\u8868\u6703\u7acb\u5373\u50b3\u5c0e\u6210\u71df\u6536\u6216\u7372\u5229\u8b8a\u5316\u3002",
                "\u82e5\u77ed\u7dda\u65b0\u805e\u4e3b\u8981\u5728\u8ac7\u4f9b\u61c9\u93c8\u5c55\u671b\u3001\u8cc7\u672c\u652f\u51fa\u6216\u984c\u6750\u60c5\u7dd2\uff0c\u4e0d\u4e00\u5b9a\u80fd\u76f4\u63a5\u4ee3\u8868\u6700\u7d42\u8a02\u55ae\u771f\u5be6\u8f49\u5f31\u3002",
                "\u7f3a\u5c11\u516c\u53f8\u516c\u544a\u3001\u8a02\u55ae\u6216\u6cd5\u8aaa\u8b49\u64da\u6642\uff0c\u5c0d\u5f71\u97ff\u5e45\u5ea6\u7684\u5224\u65b7\u61c9\u63a7\u5236\u5728\u63cf\u8ff0\u5c64\u7d1a\u3002",
            ]
        elif query.question_type == "revenue_growth_review":
            risks = [
                "營收占比若不是公司正式揭露，常來自媒體或法人估算，可能隨季度而變動。",
                "2026 年成長預測具有前瞻不確定性，容易受雲端資本支出、客戶拉貨節奏與供應鏈瓶頸影響。",
                "若現有證據主要是新聞敘事而非公司明確數字，對成長幅度的解讀應保守。",
            ]
        elif query.question_type == "season_line_margin_review":
            risks = [
                "\u8dcc\u7834\u5b63\u7dda\u4e0d\u4e00\u5b9a\u4ee3\u8868\u4e2d\u671f\u8da8\u52e2\u78ba\u8a8d\u8f49\u7a7a\uff0c\u4ecd\u9700\u89c0\u5bdf\u5f8c\u7e8c\u5e7e\u500b\u4ea4\u6613\u65e5\u662f\u5426\u7e7c\u7e8c\u5931\u5b88\u3002",
                "\u878d\u8cc7\u9918\u984d\u504f\u9ad8\u4e0d\u4ee3\u8868\u5fc5\u7136\u6703\u51fa\u73fe\u4fee\u6b63\uff0c\u4f46\u5c0d\u6ce2\u52d5\u653e\u5927\u7684\u98a8\u96aa\u78ba\u5be6\u8f03\u9ad8\u3002",
                "\u82e5\u516c\u958b\u8cc7\u6599\u7f3a\u5c11\u4e3b\u6d41\u4f86\u6e90\u76f4\u63a5\u8a55\u8ad6\u878d\u8cc7\u71b1\u5ea6\uff0c\u5c0d\u300c\u5e02\u5834\u770b\u6cd5\u300d\u7684\u63cf\u8ff0\u4ecd\u61c9\u4ee5\u7c4c\u78bc\u63a8\u4f30\u70ba\u4e3b\u3002",
            ]
        elif query.question_type == "dividend_yield_review":
            risks = [
                "\u80a1\u5229\u6700\u7d42\u5167\u5bb9\u4ecd\u9700\u4ee5\u8463\u4e8b\u6703\u3001\u80a1\u6771\u6703\u6216\u516c\u53f8\u6b63\u5f0f\u516c\u544a\u70ba\u6e96\u3002",
                "\u73fe\u91d1\u6b96\u5229\u7387\u6703\u96a8\u80a1\u50f9\u8b8a\u52d5\uff0c\u67e5\u8a62\u7576\u4e0b\u8207\u5be6\u969b\u8cb7\u5165\u6642\u9ede\u53ef\u80fd\u4e0d\u540c\u3002",
                "\u82e5\u516c\u53f8\u5f8c\u7e8c\u66f4\u65b0\u80a1\u5229\u653f\u7b56\uff0c\u73fe\u6709\u63db\u7b97\u7d50\u679c\u9700\u8981\u91cd\u65b0\u6aa2\u67e5\u3002",
            ]
        elif query.question_type == "ex_dividend_performance":
            risks = [
                "\u586b\u606f\u8868\u73fe\u53ea\u53cd\u6620\u7576\u5929\u5e02\u5834\u884c\u70ba\uff0c\u4e0d\u4ee3\u8868\u5f8c\u7e8c\u8d70\u52e2\u4e00\u5b9a\u5ef6\u7e8c\u3002",
                "\u82e5\u55ae\u65e5\u6ce2\u52d5\u53d7\u5230\u5927\u76e4\u6216\u5916\u90e8\u6d88\u606f\u5f71\u97ff\uff0c\u53ef\u80fd\u653e\u5927\u6216\u626d\u66f2\u586b\u606f\u89c0\u5bdf\u3002",
                "\u82e5\u7f3a\u5c11\u5b8c\u6574\u4ea4\u6613\u8cc7\u6599\u6216\u516c\u958b\u5831\u5c0e\uff0c\u5c0d\u5e02\u5834\u53cd\u61c9\u7684\u63cf\u8ff0\u61c9\u4fdd\u5b88\u89e3\u8b80\u3002",
            ]
        elif query.question_type in {"earnings_summary", "eps_dividend_review"}:
            risks = [
                "\u80a1\u5229\u6700\u7d42\u5167\u5bb9\u4ecd\u9700\u4ee5\u8463\u4e8b\u6703\u3001\u80a1\u6771\u6703\u6216\u516c\u53f8\u6b63\u5f0f\u516c\u544a\u70ba\u6e96\u3002",
                "\u6b77\u53f2 EPS \u4e0d\u80fd\u76f4\u63a5\u4fdd\u8b49\u672a\u4f86\u7372\u5229\u5ef6\u7e8c\u3002",
                "\u82e5\u5e02\u5834\u9810\u671f\u4e3b\u8981\u4f86\u81ea\u55ae\u4e00\u65b0\u805e\u4f86\u6e90\uff0c\u89e3\u8b80\u4e0a\u8981\u4fdd\u7559\u5f48\u6027\u3002",
            ]
        elif query.question_type == "announcement_summary":
            risks = [
                "\u516c\u544a\u6a19\u984c\u5bb9\u6613\u88ab\u5feb\u901f\u89e3\u8b80\uff0c\u4ecd\u9700\u56de\u770b\u5b8c\u6574\u63ed\u9732\u5167\u5bb9\u3002",
                "\u82e5\u53ea\u6709\u55ae\u4e00\u516c\u544a\u800c\u7f3a\u4e4f\u4ea4\u53c9\u4f86\u6e90\uff0c\u5e02\u5834\u89e3\u8b80\u53ef\u80fd\u504f\u5dee\u3002",
                "\u516c\u544a\u5c0d\u80a1\u50f9\u7684\u5f71\u97ff\u5e38\u53d7\u6642\u9593\u9ede\u8207\u5e02\u5834\u60c5\u7dd2\u653e\u5927\u6216\u920d\u5316\u3002",
            ]
        elif is_fundamental_valuation_question(query):
            risks = [
                "\u4f30\u503c\u9ad8\u4f4e\u6703\u96a8\u80a1\u50f9\u8207\u7372\u5229\u9810\u671f\u8b8a\u5316\uff0c\u4eca\u5929\u7684\u672c\u76ca\u6bd4\u4f4d\u7f6e\u4e0d\u4ee3\u8868\u5f8c\u7e8c\u4e0d\u6703\u518d\u8abf\u6574\u3002",
                "\u57fa\u672c\u9762\u5982\u679c\u53ea\u6709\u55ae\u4e00\u5b63\u6216\u55ae\u6708\u8b49\u64da\uff0c\u4e0d\u4e00\u5b9a\u80fd\u4ee3\u8868\u4e2d\u9577\u671f\u7372\u5229\u8da8\u52e2\u3002",
                "\u55ae\u770b\u672c\u76ca\u6bd4\u6216\u55ae\u770b\u71df\u6536\u90fd\u53ef\u80fd\u5931\u771f\uff0c\u4ecd\u8981\u642d\u914d\u73fe\u91d1\u6d41\u3001\u8cc7\u672c\u652f\u51fa\u548c\u7522\u696d\u666f\u6c23\u4e00\u8d77\u89e3\u8b80\u3002",
            ]
        elif query.question_type == "price_outlook":
            risks = [
                "\u6f32\u8dcc\u9810\u6e2c\u672c\u8eab\u5177\u6709\u9ad8\u5ea6\u4e0d\u78ba\u5b9a\u6027\uff0c\u4e0d\u80fd\u8996\u70ba\u78ba\u5b9a\u65b9\u5411\u3002",
                "\u77ed\u671f\u50f9\u683c\u53ef\u80fd\u53d7\u5230\u5916\u90e8\u6d88\u606f\u3001\u8cc7\u91d1\u8f2a\u52d5\u8207\u6574\u9ad4\u5e02\u5834\u98a8\u96aa\u5f71\u97ff\u3002",
                "\u82e5\u7f3a\u5c11\u8fd1\u671f\u65b0\u805e\u8207\u516c\u544a\u4f50\u8b49\uff0c\u4efb\u4f55\u65b9\u5411\u5224\u65b7\u90fd\u61c9\u4fdd\u5b88\u770b\u5f85\u3002",
            ]
        elif query.stance_bias != StanceBias.NEUTRAL:
            risks = [
                "\u7576\u554f\u984c\u5e36\u6709\u55ae\u4e00\u7acb\u5834\u6642\uff0c\u5bb9\u6613\u5ffd\u7565\u76f8\u53cd\u8b49\u64da\u3002",
                "\u82e5\u53ea\u6311\u9078\u652f\u6301\u65e2\u6709\u770b\u6cd5\u7684\u8cc7\u8a0a\uff0c\u5224\u8b80\u504f\u8aa4\u6703\u88ab\u653e\u5927\u3002",
                "\u6295\u8cc7\u5224\u65b7\u4ecd\u61c9\u540c\u6642\u6aa2\u67e5\u5229\u591a\u3001\u5229\u7a7a\u8207\u4e0d\u78ba\u5b9a\u56e0\u7d20\u3002",
            ]
        else:
            risks = [
                "\u82e5\u8cc7\u6599\u4e3b\u8981\u96c6\u4e2d\u5728\u55ae\u4e00\u6642\u9593\u7a97\uff0c\u53ef\u80fd\u5ffd\u7565\u66f4\u9577\u671f\u7684\u57fa\u672c\u9762\u8b8a\u5316\u3002",
                "\u82e5\u5f8c\u7e8c\u51fa\u73fe\u66f4\u65b0\u516c\u544a\uff0c\u73fe\u6709\u6458\u8981\u9700\u8981\u91cd\u65b0\u9a57\u8b49\u3002",
                "\u4e0d\u540c\u4f86\u6e90\u5c0d\u540c\u4e00\u4e8b\u4ef6\u7684\u89e3\u8b80\u53ef\u80fd\u4e0d\u540c\uff0c\u9700\u6301\u7e8c\u6bd4\u5c0d\u3002",
            ]

        if governance_report.high_trust_ratio < 0.5:
            risks.append("\u76ee\u524d\u9ad8\u53ef\u4fe1\u4f86\u6e90\u5360\u6bd4\u4e0d\u9ad8\uff0c\u5efa\u8b70\u56de\u770b\u539f\u6587\u518d\u505a\u5224\u65b7\u3002")

        return risks[:4]

    def _extract_price_range(self, governance_report: GovernanceReport) -> tuple[str | None, str | None]:
        high_pattern = re.compile(r"\u6700\u9ad8\u50f9(?:\u70ba)?\s*(\d+(?:\.\d+)?)\s*\u5143")
        low_pattern = re.compile(r"\u6700\u4f4e\u50f9(?:\u70ba)?\s*(\d+(?:\.\d+)?)\s*\u5143")
        for evidence in governance_report.evidence:
            high_match = high_pattern.search(evidence.excerpt)
            low_match = low_pattern.search(evidence.excerpt)
            if high_match and low_match:
                return high_match.group(1), low_match.group(1)
        return None, None

    def _extract_company_margin(self, governance_report: GovernanceReport, company_name: str) -> str | None:
        pattern = re.compile(rf"{re.escape(company_name)}[^。；]*?\u6bdb\u5229\u7387\u7d04 (\d+(?:\.\d+)?)%")
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
        if "轉上市" in query.user_query:
            return "轉上市後"
        if "上市" in query.user_query:
            return "上市後"
        return "掛牌後"
