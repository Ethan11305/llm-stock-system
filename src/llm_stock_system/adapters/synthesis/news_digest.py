from llm_stock_system.core.enums import TopicTag
from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import (
    build_risks_generic,
    build_summary_fallback,
    evidence_contains,
    listing_event_label,
    extract_number,
    extract_text,
)

_THEME_TAGS: frozenset[str] = frozenset({
    "題材", "產業",
    TopicTag.EV.value,
    TopicTag.AI.value,
    TopicTag.SEMICON_EQUIP.value,
})


class NewsDigestStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.SHIPPING.value in tags or "SCFI" in tags:
            return self._summarize_shipping(label, query, report)
        if TopicTag.ELECTRICITY.value in tags:
            return self._summarize_electricity(label, query, report)
        if TopicTag.MACRO.value in tags or "殖利率" in tags or "CPI" in tags:
            return self._summarize_macro_yield(label, report)
        if tags & _THEME_TAGS:
            return self._summarize_theme_impact(label, query, report)
        if TopicTag.GUIDANCE.value in tags or "指引" in tags:
            return self._summarize_guidance_reaction(label, report)
        if TopicTag.LISTING.value in tags:
            return self._summarize_listing_revenue(label, query, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
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
        return [
            "新聞能幫助理解市場目前關注的事件與敘事。",
            "若多個來源集中報導同一主題，代表關注度可能升高。",
            "仍需回看原文與公告，避免只看摘要造成誤讀。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
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
        return build_risks_generic(query)

    # ── private sub-summarizers ─────────────────────────────────────────

    def _summarize_shipping(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        shipping_label = label if not comparison_label else f"{label}與{comparison_label}"
        has_red_sea = evidence_contains(report, ("紅海", "紅海航線", "繞道", "受阻"))
        has_scfi = evidence_contains(report, ("SCFI", "運價指數", "運價"))
        has_target_price = evidence_contains(report, ("目標價", "評等", "分析師", "法人", "外資"))
        has_divergent_target = evidence_contains(report, ("分歧", "正負解讀並存", "方向尚未完全一致"))
        has_positive_target = evidence_contains(report, ("上修", "調高", "買進", "看好", "受惠", "有戲"))
        has_negative_target = evidence_contains(report, ("下修", "調降", "保守", "觀望", "中立", "壓力"))
        if report.evidence:
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
                return (
                    f"{shipping_label} 目前可整理到的最新訊息顯示，"
                    + "；".join(points)
                    + "；後續仍要看 SCFI 續航與航線壅塞是否延續。"
                )
            return (
                f"{shipping_label} 目前已有部分航運事件新聞可供整理，"
                "但公開資訊仍不足以穩定判斷 SCFI 支撐力道與目標價調整方向。"
            )
        return f"資料不足，無法確認紅海航線受阻對{shipping_label}的 SCFI 支撐力道與分析師目標價調整。"

    def _summarize_electricity(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        electricity_label = label if not comparison_label else f"{label}與{comparison_label}"
        has_tariff = evidence_contains(report, ("工業電價", "電價", "調漲", "漲價", "電費"))
        has_cost = evidence_contains(report, ("成本", "毛利", "壓力", "費用", "獲利"))
        has_amount = evidence_contains(report, ("億元", "千萬元", "增加額度", "增幅", "%"))
        has_response = evidence_contains(
            report, ("因應", "對策", "節能", "節電", "降耗", "轉嫁", "調價", "綠電", "自發電")
        )
        if report.evidence:
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

    def _summarize_macro_yield(self, label: str, report: GovernanceReport) -> str:
        has_cpi = evidence_contains(report, ("CPI", "通膨"))
        has_rate = evidence_contains(report, ("利率", "美債", "殖利率", "高殖利率"))
        has_financial_sector = evidence_contains(report, ("金控", "金控股", "金融股"))
        has_negative = evidence_contains(report, ("負面", "保守", "觀望", "壓力", "降溫"))
        has_defensive = evidence_contains(report, ("防禦", "現金流", "穩健", "支撐"))
        has_institutional_view = evidence_contains(report, ("法人", "外資", "觀點", "看法"))
        if report.evidence:
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
            return (
                f"{label} 目前已有部分總經與高殖利率題材新聞可供整理，"
                "但仍不足以穩定判斷 CPI 對高殖利率股與金控股情緒的傳導幅度。"
            )
        return f"資料不足，無法確認美國 CPI 對{label}與高殖利率族群的負面情緒影響與法人觀點。"

    def _summarize_theme_impact(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        comparison_label = query.comparison_company_name or query.comparison_ticker
        theme_label = label if not comparison_label else f"{label}、{comparison_label}"
        has_cobalt_signal = evidence_contains(report, ("鈷價", "鈷", "油價", "原料", "cobalt"))
        has_energy_transition = evidence_contains(report, ("能源轉型", "城市採礦"))
        has_product_mix = evidence_contains(report, ("產品組合", "庫存效應", "產能利用率"))
        has_asml_signal = evidence_contains(
            report, ("ASML", "艾司摩爾", "展望", "不如預期", "保守", "下修")
        )
        has_equipment_signal = evidence_contains(
            report, ("半導體設備", "設備", "EUV", "曝光機", "資本支出", "擴產", "訂單")
        )
        has_sentiment_signal = evidence_contains(
            report, ("利空", "情緒", "觀望", "保守", "降溫", "下修")
        )
        if report.evidence:
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
            return (
                f"{theme_label} 目前有部分新聞可供參考，"
                "但公開資訊仍不足以直接量化 ASML 展望變化對設備族群基本面的實際傳導強度。"
            )
        return f"資料不足，無法確認ASML展望轉弱對{theme_label}的最新利空分析與情緒影響。"

    def _summarize_guidance_reaction(self, label: str, report: GovernanceReport) -> str:
        positive_count = extract_number(report, r"正面解讀共有 (\d+) 則")
        negative_count = extract_number(report, r"負面解讀共有 (\d+) 則")
        has_positive = positive_count is not None or evidence_contains(
            report, ("正面反應整理", "看好", "調高", "上修")
        )
        has_negative = negative_count is not None or evidence_contains(
            report, ("負面反應整理", "保守", "下修", "不如預期")
        )
        if has_positive and has_negative:
            return (
                f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正負並存；"
                "目前可整理到同時存在正面與負面反應，顯示市場對後續動能仍有分歧。"
            )
        if has_positive:
            return (
                f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向正面；"
                "現有中高可信來源較多聚焦於成長動能或展望支撐。"
            )
        if has_negative:
            return (
                f"{label} 法說會後，媒體與法人對下半年營運指引的解讀偏向保守；"
                "現有來源較多聚焦於下修、壓力或不確定性。"
            )
        return f"資料不足，無法確認{label}法說會後媒體與法人對下半年營運指引的正負面反應。"

    def _summarize_listing_revenue(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        event_label = listing_event_label(query)
        revenue_month = extract_text(report, r"最新已公布月營收為 (\d{4}-\d{2})")
        month_revenue = extract_number(report, r"單月營收約 (\d+(?:\.\d+)?) 億元")
        mom_pct = extract_number(report, r"月增率約 (-?\d+(?:\.\d+)?)%")
        yoy_pct = extract_number(report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        has_route_news = evidence_contains(report, ("航線", "熊本", "旅展", "機票"))
        has_fee_news = evidence_contains(report, ("燃油附加費", "票價", "附加費"))
        has_brand_news = evidence_contains(report, ("A350", "空山基", "機上餐飲", "品牌"))
        if revenue_month and month_revenue:
            parts = [f"{label} {event_label}的股價波動，現有資料較像市場對營運題材與消息面交互反應。"]
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
        return f"資料不足，無法確認{label}{event_label}股價波動的主因，以及是否有重大營收增長消息。"
