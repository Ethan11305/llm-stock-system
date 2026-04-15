# Phase 4: GenerationLayer 重構
## 以 Intent + topic_tags 取代 question_type 分支

**Status**: Design — Ready for Implementation  
**Estimated Effort**: 3–4 days  
**Depends on**: Phase 3 (ValidationLayer facet-based scoring 完成)  
**Risk Level**: Medium（觸及兩個 LLM adapter，但 GenerationLayer 本身已乾淨）

---

## 1. 現狀診斷

### 1.1 GenerationLayer 本身已乾淨

`layers/generation_layer.py` 本身只有 18 行，完全無 `question_type` 路由：

```python
class GenerationLayer:
    def generate(self, query, governance_report) -> AnswerDraft:
        system_prompt = self._load_prompt()
        return self._llm_client.synthesize(query, governance_report, system_prompt)
```

問題全部在它呼叫的兩個 LLM adapter 裡。

### 1.2 `adapters/llm.py`（RuleBasedSynthesisClient）— 嚴重的 question_type 耦合

`_build_summary()` 有 **20+ 個** `if query.question_type == "..."` 硬編碼分支，涵蓋：

| question_type | 現狀 |
|---|---|
| `margin_turnaround_review` | 獨立 regex 抽取邏輯 |
| `listing_revenue_review` | 獨立敘述邏輯 |
| `guidance_reaction_review` | 正/負計數邏輯 |
| `monthly_revenue_yoy_review` | 單月+累計雙路徑 |
| `price_range` | 高低價抽取 |
| `gross_margin_comparison_review` | 雙標的比較 |
| `profitability_stability_review` | 近五年穩定性 |
| `debt_dividend_safety_review` | 負債比+現金覆蓋 |
| `fcf_dividend_sustainability_review` | FCF 對比股利 |
| `pe_valuation_review` | 本益比區間 |
| `price_outlook` | 目標價/方向 |
| `revenue_growth_review` | AI 伺服器營收 |
| `technical_indicator_review` | RSI/KD/MACD |
| `season_line_margin_review` | 季線+籌碼 |
| `shipping_rate_impact_review` | 紅海/SCFI/目標價 |
| `electricity_cost_impact_review` | 電價/因應措施 |
| `macro_yield_sentiment_review` | CPI/殖利率/法人 |
| `theme_impact_review` | ASML/半導體設備/主題 |
| `dividend_yield_review` | 現金股利+殖利率 |
| `ex_dividend_performance` | 除息日+填息率 |
| `eps_dividend_review` | EPS+股利 |
| `earnings_summary` | 財報摘要 |
| `announcement_summary` | 最新公告 |

`_build_impacts()` 與 `_build_risks()` 也有多個 `question_type` 分支。

### 1.3 `adapters/openai_responses.py`（OpenAIResponsesSynthesisClient）— 雙重耦合

**User Prompt 組裝**（`synthesize()` 與 `_synthesize_preliminary()`）都有：

```python
f"Question type: {query.question_type}",
```

Intent 與 topic_tags 完全沒有傳入 LLM，代表 LLM 根本看不到重構後的語義路由。

**Guardrails** 也直接用 question_type：

```python
def _apply_target_price_guardrails(self, ...):
    if query.question_type != "price_outlook" or not is_forward_price_question(query):
        return parsed  # ← question_type 路由
```

### 1.4 Intent 基礎設施已就緒（Phase 0 完成）

`core/enums.py` 中已定義 7 個 Intent：

```python
class Intent(str, Enum):
    NEWS_DIGEST          = "news_digest"
    EARNINGS_REVIEW      = "earnings_review"
    VALUATION_CHECK      = "valuation_check"
    DIVIDEND_ANALYSIS    = "dividend_analysis"
    FINANCIAL_HEALTH     = "financial_health"
    TECHNICAL_VIEW       = "technical_view"
    INVESTMENT_ASSESSMENT = "investment_assessment"
```

`StructuredQuery` 已有 `intent`、`controlled_tags`、`topic_tags` 欄位，且 `QUESTION_TYPE_TO_INTENT` 映射已建立。

---

## 2. Phase 4 目標

1. `llm.py` 的 20+ `question_type` 分支重寫為 **7 個 Intent template 方法**，以 `intent` 為一級路由，`controlled_tags`/`topic_tags` 為二級精化。
2. `openai_responses.py` 的 user prompt 組裝邏輯改為以 `intent` + `controlled_tags` 決定強調重點，`question_type` 不再出現在傳送給 LLM 的 prompt 中。
3. Phase 4 完成後，`question_type` 在 GenerationLayer 這條路徑上**完全不作為路由依據**（解耦完成）。

> **注意**：`question_type` 欄位本身可以繼續存在於 `StructuredQuery`，以供 InputLayer 向下相容，但 GenerationLayer 之後的任何程式不應再以它做分支。

---

## 3. 核心設計

### 3.1 Intent → Template 對應關係

每個 Intent 對應一個 `_build_summary_*` 方法。現有 `question_type` 的邏輯依 `QUESTION_TYPE_TO_INTENT` 對應關係合併：

| Intent | 原 question_type（合併進來） | 二級精化方式（由 controlled_tags 決定） |
|--------|---------------------------|--------------------------------------|
| `NEWS_DIGEST` | `shipping_rate_impact_review`, `electricity_cost_impact_review`, `macro_yield_sentiment_review`, `guidance_reaction_review`, `listing_revenue_review`, `theme_impact_review`, `market_summary` | tags: `航運`, `電價`, `總經`, `法說`, `上市` |
| `EARNINGS_REVIEW` | `earnings_summary`, `eps_dividend_review`, `monthly_revenue_yoy_review`, `margin_turnaround_review` | tags: `月營收`, `毛利率` → 進入對應子邏輯 |
| `VALUATION_CHECK` | `pe_valuation_review`, `fundamental_pe_review`, `price_range`, `price_outlook` | tags: `本益比`, `基本面` |
| `DIVIDEND_ANALYSIS` | `dividend_yield_review`, `ex_dividend_performance`, `fcf_dividend_sustainability_review`, `debt_dividend_safety_review` | tags: `股利`, `現金流`, `負債` |
| `FINANCIAL_HEALTH` | `profitability_stability_review`, `gross_margin_comparison_review`, `revenue_growth_review` | tags: `毛利率`, `基本面` |
| `TECHNICAL_VIEW` | `technical_indicator_review`, `season_line_margin_review` | tags: `技術面`, `籌碼` |
| `INVESTMENT_ASSESSMENT` | `investment_support`, `risk_review`, `announcement_summary` | tags: `基本面`, `本益比` |

### 3.2 二級精化原則（controlled_tags 驅動）

在每個 Intent template 方法內部，**不再用 `question_type` 分支**，改用 tag 集合判斷：

```python
def _build_summary_news_digest(self, query, governance_report) -> str:
    tags = set(query.topic_tags)       # 已含 controlled_tags + free_keywords 合併後的值
    label = query.company_name or query.ticker or "此標的"

    # 以 tag 決定進入哪個子邏輯
    if "航運" in tags:
        return self._summarize_shipping_context(label, query, governance_report)
    if "電價" in tags:
        return self._summarize_electricity_context(label, query, governance_report)
    if "總經" in tags:
        return self._summarize_macro_yield_context(label, query, governance_report)
    if "法說" in tags:
        return self._summarize_guidance_reaction_context(label, query, governance_report)
    if "上市" in tags:
        return self._summarize_listing_revenue_context(label, query, governance_report)

    # 無特定 tag → 通用新聞摘要
    return self._summarize_generic_news(label, governance_report)
```

> 這些 `_summarize_*_context()` 方法的實作內容與現有 `question_type` 分支的邏輯完全相同，**只是搬移**，不改邏輯。

### 3.3 `_build_summary()` 主路由改造

```python
# 改造前
def _build_summary(self, query, governance_report) -> str:
    if query.question_type == "margin_turnaround_review": ...
    if query.question_type == "listing_revenue_review": ...
    # ... 20+ 分支

# 改造後
_INTENT_SUMMARY_BUILDERS = {
    Intent.NEWS_DIGEST:           "_build_summary_news_digest",
    Intent.EARNINGS_REVIEW:       "_build_summary_earnings_review",
    Intent.VALUATION_CHECK:       "_build_summary_valuation_check",
    Intent.DIVIDEND_ANALYSIS:     "_build_summary_dividend_analysis",
    Intent.FINANCIAL_HEALTH:      "_build_summary_financial_health",
    Intent.TECHNICAL_VIEW:        "_build_summary_technical_view",
    Intent.INVESTMENT_ASSESSMENT: "_build_summary_investment_assessment",
}

def _build_summary(self, query, governance_report) -> str:
    builder_name = self._INTENT_SUMMARY_BUILDERS.get(query.intent)
    if builder_name:
        builder = getattr(self, builder_name)
        return builder(query, governance_report)
    return self._build_summary_fallback(query, governance_report)
```

### 3.4 `openai_responses.py` 的 User Prompt 改造

**改造前**（current）：

```python
user_prompt = "\n".join([
    f"User query: {query.user_query}",
    f"Ticker: {query.ticker or 'unknown'}",
    f"Company: {query.company_name or 'unknown'}",
    f"Comparison ticker: {query.comparison_ticker or 'none'}",
    f"Comparison company: {query.comparison_company_name or 'none'}",
    f"Topic: {query.topic.value}",
    f"Question type: {query.question_type}",   # ← 移除
    "Evidence:",
    ...
])
```

**改造後**：

```python
user_prompt = "\n".join([
    f"User query: {query.user_query}",
    f"Ticker: {query.ticker or 'unknown'}",
    f"Company: {query.company_name or 'unknown'}",
    f"Comparison ticker: {query.comparison_ticker or 'none'}",
    f"Comparison company: {query.comparison_company_name or 'none'}",
    f"Intent: {query.intent.value}",                          # ← 新增
    f"Topic tags: {', '.join(query.topic_tags) or 'none'}",   # ← 新增
    *self._build_intent_instructions(query),                   # ← 新增：Intent-specific hints
    "Evidence:",
    ...
])
```

### 3.5 7 個 Intent 的 User Prompt Instructions

新增私有方法 `_build_intent_instructions(query) -> list[str]`，依 intent 返回對應的強調說明：

```python
_INTENT_INSTRUCTIONS: dict[Intent, list[str]] = {
    Intent.NEWS_DIGEST: [
        "Focus: summarize the market sentiment and key event triggers from the evidence.",
        "If topic tags include '航運', emphasize freight-rate signals and analyst target-price reactions.",
        "If topic tags include '電價', emphasize electricity cost pressure and company response measures.",
        "If topic tags include '總經', emphasize macro indicators (CPI, yield) and institutional views.",
        "If topic tags include '法說', summarize positive/negative interpretations of the guidance.",
        "If topic tags include '上市', connect revenue data with post-listing price movement.",
    ],
    Intent.EARNINGS_REVIEW: [
        "Focus: summarize earnings quality, trend, and momentum from the financial evidence.",
        "If topic tags include '月營收', highlight MoM/YoY revenue change and whether it hit a 12-month high.",
        "If topic tags include '毛利率', explicitly address whether gross margin has turned positive and whether operating income followed.",
    ],
    Intent.VALUATION_CHECK: [
        "Focus: assess current valuation relative to historical range and peers.",
        "State the current PE ratio and its historical percentile position.",
        "If topic tags include '基本面', combine PE with fundamental quality indicators.",
        "For forward price questions, anchor to analyst target prices or price-level evidence; do not fabricate numbers.",
    ],
    Intent.DIVIDEND_ANALYSIS: [
        "Focus: assess dividend yield, payout sustainability, and coverage ratios.",
        "If topic tags include '現金流', address whether free cash flow covers dividend payments over the past 3 years.",
        "If topic tags include '負債', address debt ratio trend and cash balance coverage of dividend obligations.",
    ],
    Intent.FINANCIAL_HEALTH: [
        "Focus: assess profitability stability, margin structure, and revenue growth.",
        "If a comparison ticker is present, compare gross margins side-by-side using the most recent comparable period.",
        "Mention any loss years in the past 5 years and their likely cause.",
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

def _build_intent_instructions(self, query: StructuredQuery) -> list[str]:
    base = self._INTENT_INSTRUCTIONS.get(query.intent, [])
    tag_hints = [
        f"Active topic tags: {', '.join(query.topic_tags)}" if query.topic_tags else ""
    ]
    return [line for line in tag_hints + base if line]
```

### 3.6 Guardrails 改為 Intent + tag 路由

**改造前**：

```python
def _apply_target_price_guardrails(self, query, governance_report, parsed):
    if query.question_type != "price_outlook" or not is_forward_price_question(query):
        return parsed
    ...
```

**改造後**：

```python
def _apply_target_price_guardrails(self, query, governance_report, parsed):
    # 條件改為：intent 是 VALUATION_CHECK 且確認為前瞻價格問題
    if query.intent != Intent.VALUATION_CHECK or not is_forward_price_question(query):
        return parsed
    ...
```

> `is_forward_price_question()` 內部的邏輯（檢查 user_query 關鍵字）不在 Phase 4 修改範圍。

---

## 4. 修改範圍

### 4.1 需修改的檔案

| 檔案 | 修改內容 | 風險 |
|------|----------|------|
| `adapters/llm.py` | `_build_summary()` 主路由改為 Intent dispatch；現有 20+ 分支內容搬移為 7 個 intent method + tag-driven sub-methods | 中（需精確搬移，邏輯不變） |
| `adapters/openai_responses.py` | user prompt 移除 `question_type`，改加 `intent`、`topic_tags`、intent instructions；guardrails 改用 intent 判斷；`_synthesize_preliminary()` 同步更新 | 中 |
| `prompts/system_prompt.md` | 新增一節「Intent-specific guidance」，說明 7 個 intent 的輸出重點 | 低 |

### 4.2 不需修改的檔案

| 檔案 | 原因 |
|------|------|
| `layers/generation_layer.py` | 已完全乾淨，無 `question_type` |
| `core/enums.py` | `Intent`、`TopicTag` 已完整（Phase 0） |
| `core/models.py` | `StructuredQuery` 結構已完整（Phase 0），`QUESTION_TYPE_TO_INTENT` 映射已就緒 |
| `layers/validation_layer.py` | Phase 3 已完成，Group A no-op 分支將在 Phase 4 後可以刪除（但 Phase 4 本身不刪） |
| `core/fundamental_valuation.py` | 內部邏輯不動，但它的 `is_fundamental_valuation_question()` 可能需要改為 intent-based（見 §5.1） |
| `core/target_price.py` | `is_forward_price_question()` 內部邏輯不動 |
| `orchestrator/pipeline.py` | 不需修改 |

### 4.3 新增測試檔案

- `tests/test_generation_layer_phase4.py`

---

## 5. 邊界情況與注意事項

### 5.1 `is_fundamental_valuation_question()` 的問題

`core/fundamental_valuation.py` 裡的 `is_fundamental_valuation_question()` 目前是：

```python
def is_fundamental_valuation_question(query: StructuredQuery) -> bool:
    return query.question_type in {"fundamental_pe_review", "investment_support"}
```

這個函式被 `llm.py` 與 `openai_responses.py` 都呼叫。Phase 4 應將它改為 intent-based：

```python
def is_fundamental_valuation_question(query: StructuredQuery) -> bool:
    from llm_stock_system.core.enums import Intent, TopicTag
    if query.intent == Intent.VALUATION_CHECK:
        return TopicTag.FUNDAMENTAL.value in query.topic_tags or TopicTag.VALUATION.value in query.topic_tags
    if query.intent == Intent.INVESTMENT_ASSESSMENT:
        return True
    # 向下相容：若 intent 為預設但 question_type 仍在舊值
    return query.question_type in {"fundamental_pe_review", "investment_support"}
```

> 這個改動風險低，因為 `QUESTION_TYPE_TO_INTENT` 映射已確保 `intent` 被正確填入。

### 5.2 `_build_impacts()` 和 `_build_risks()` 的處理策略

與 `_build_summary()` 相同，改為 7 個 intent 路由。

- **impacts**：每個 intent 有一套通用 impacts，`topic_tags` 可以覆蓋特定文字。
- **risks**：每個 intent 有三條通用風險提醒；`INVESTMENT_ASSESSMENT` 的 risks 要求至少三條特定內容。

### 5.3 `RuleBasedSynthesisClient` 的向下相容

在所有 intent method 完成之前，可以在每個 intent method 末尾保留一個 fallback：

```python
def _build_summary_news_digest(self, query, governance_report) -> str:
    tags = set(query.topic_tags)
    label = query.company_name or query.ticker or "此標的"

    if "航運" in tags:
        return self._summarize_shipping_context(label, query, governance_report)
    # ... 其他 tags
    
    # Fallback: 通用新聞摘要
    official_count = sum(1 for item in governance_report.evidence if item.source_tier == SourceTier.HIGH)
    if official_count:
        return f"{label} 目前有官方或高可信來源可供整理，但現有資訊仍應與後續公告一併觀察。"
    return f"{label} 目前有部分可參考資料，但來源強度與一致性仍需持續確認。"
```

### 5.4 `question_type` 欄位的最終狀態

Phase 4 完成後：
- `StructuredQuery.question_type` 欄位**繼續保留**（向下相容 InputLayer / API）
- GenerationLayer 的所有程式碼**不讀取** `question_type`
- ValidationLayer 的 Group A no-op 分支（只剩 warning，不 cap）可以在 **Phase 5** 清除
- `QUESTION_TYPE_TO_INTENT` 映射繼續作為 InputLayer → StructuredQuery 建構時的自動轉換工具

---

## 6. 實作執行順序

```
Step 1: core/fundamental_valuation.py
    → is_fundamental_valuation_question() 改為 intent-based（含向下相容）

Step 2: adapters/llm.py — 分三個子步驟
    Step 2a: 新增 _INTENT_SUMMARY_BUILDERS dispatch table
    Step 2b: 建立 7 個 _build_summary_<intent>() 方法骨架
    Step 2c: 逐一把現有 question_type 分支內容搬移至對應的 sub-method
             （順序建議：NEWS_DIGEST → EARNINGS_REVIEW → VALUATION_CHECK
               → DIVIDEND_ANALYSIS → FINANCIAL_HEALTH → TECHNICAL_VIEW
               → INVESTMENT_ASSESSMENT）
    Step 2d: 同樣重構 _build_impacts() 與 _build_risks()

Step 3: adapters/openai_responses.py
    Step 3a: 新增 _INTENT_INSTRUCTIONS dict 與 _build_intent_instructions() 方法
    Step 3b: 修改 synthesize() 的 user_prompt 組裝（移除 question_type，加入 intent+tags+instructions）
    Step 3c: 修改 _synthesize_preliminary() 的 user_prompt（同步移除 question_type）
    Step 3d: 修改 _apply_target_price_guardrails()（改為 intent 判斷）

Step 4: prompts/system_prompt.md
    → 新增 "Intent guidance" 節，說明各 Intent 的輸出重點（選配，可強化 LLM 效果）

Step 5: tests/test_generation_layer_phase4.py
    → 依照第 7 節的 case 逐一新增

Step 6: pytest 全回歸
    → 確認 Phase 3 後的所有測試全過
```

---

## 7. 測試策略

### 7.1 新增測試（`test_generation_layer_phase4.py`）

#### Case 1：NEWS_DIGEST + tag=航運 → 路由到航運邏輯

```python
# query.intent = Intent.NEWS_DIGEST
# query.controlled_tags = [TopicTag.SHIPPING]
# governance_report 含紅海/SCFI 相關 evidence
# 期待：summary 包含"紅海"或"SCFI"相關文字，無任何 question_type 路由痕跡
```

#### Case 2：EARNINGS_REVIEW + tag=毛利率 → 路由到 margin_turnaround 邏輯

```python
# query.intent = Intent.EARNINGS_REVIEW
# query.controlled_tags = [TopicTag.GROSS_MARGIN]
# governance_report 含毛利率/營業利益 evidence
# 期待：summary 正確抽取毛利率數字，不依賴 question_type
```

#### Case 3：VALUATION_CHECK + is_forward_price_question=True → guardrails 觸發

```python
# query.intent = Intent.VALUATION_CHECK
# query.user_query = "台積電明年目標價多少"
# 期待：_apply_target_price_guardrails 觸發，summary 被替換為 build_forward_price_summary()
```

#### Case 4：DIVIDEND_ANALYSIS + tag=現金流 → FCF 永續性邏輯

```python
# query.intent = Intent.DIVIDEND_ANALYSIS
# query.controlled_tags = [TopicTag.CASH_FLOW]
# 期待：summary 正確描述 FCF vs 股利支付比較
```

#### Case 5：TECHNICAL_VIEW + tag=籌碼 → 季線+籌碼邏輯

```python
# query.intent = Intent.TECHNICAL_VIEW
# query.controlled_tags = [TopicTag.MARGIN_FLOW]
# 期待：summary 包含季線(MA60)與融資餘額資訊
```

#### Case 6：openai_responses.py user prompt 不再含 question_type

```python
# 測試 _build_payload() 產生的 payload，確認：
# - "question_type" 字串不出現在 user content 中
# - "Intent:" 出現在 user content 中
# - "Topic tags:" 出現在 user content 中
```

#### Case 7：向下相容 — 舊 question_type 查詢仍能正確路由

```python
# 模擬從 InputLayer 傳入 question_type="margin_turnaround_review" 的 query
# 確認 StructuredQuery 自動填入 intent=EARNINGS_REVIEW
# 確認 RuleBasedSynthesisClient 正確路由到 EARNINGS_REVIEW + 毛利率邏輯
# （不需要任何 question_type 讀取）
```

### 7.2 回歸測試

```bash
pytest tests/ -v --tb=short
```

重點確認：
- `test_generation_layer_phase4.py`（新）
- `test_validation_layer_phase3.py`（確認 Phase 3 未被影響）
- `test_openai_preliminary_answers.py`（確認 preliminary 流程正常）
- 所有 `test_*_queries.py`（確認各場景行為未意外改變）

---

## 8. Definition of Done

進入 Phase 5 之前，以下條件必須全部成立：

- [ ] `is_fundamental_valuation_question()` 改為 intent-based，向下相容 question_type 仍有效
- [ ] `RuleBasedSynthesisClient._build_summary()` 主路由改為 7-Intent dispatch，不再含任何 `query.question_type ==` 判斷
- [ ] 7 個 `_build_summary_<intent>()` 方法完整實作，各方法以 `topic_tags` 做二級精化
- [ ] `_build_impacts()` 與 `_build_risks()` 同樣改為 intent-based 路由
- [ ] `OpenAIResponsesSynthesisClient.synthesize()` 的 user_prompt 不含 `question_type`，改含 `intent`、`topic_tags` 與 intent instructions
- [ ] `_synthesize_preliminary()` 的 user_prompt 同步更新
- [ ] `_apply_target_price_guardrails()` 改為以 `intent == Intent.VALUATION_CHECK` 判斷
- [ ] `system_prompt.md` 新增 intent guidance 節（建議，非必要）
- [ ] `tests/test_generation_layer_phase4.py` 的所有 case 通過
- [ ] 全部回歸測試通過（pytest tests/ 無 failure）

---

## 9. 後續 Phase 預告

Phase 4 完成後：

- **Phase 5**：LLM-driven generation 全面取代 `RuleBasedSynthesisClient` 的規則邏輯，`system_prompt.md` 擴充為完整的 7-intent template prompt，`RuleBasedSynthesisClient` 降為純後備用途。
- **清理工作**：`ValidationLayer` 的 Group A no-op 分支（`question_type` 判斷）可以在 Phase 5 清除；`StructuredQuery.question_type` 欄位可以在所有 InputLayer 客戶端完成遷移後標記為 deprecated。
