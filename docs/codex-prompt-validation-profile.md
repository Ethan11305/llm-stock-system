# Codex Prompt: ValidationProfile (信心評分組態) 設計與實作

## 背景脈絡

這是一個台股 LLM 諮詢系統（`src/llm_stock_system/`），採用六層 pipeline 架構：Input → Retrieval → DataGovernance → Generation → **Validation** → Presentation。

`ValidationLayer`（`layers/validation_layer.py`）負責對 LLM 生成的回答計算信心分數（0.0–1.0），並映射為紅黃綠燈號。目前 `_apply_question_type_rules()` 方法內有 17 個 if 分支，每個分支針對一種 `question_type` 做硬編碼的證據檢查與分數 cap。

Phase 3 已完成：把 14 個分支拆成三群（Group A: warning-only、Group B: 三段式 cap、Group C: 保留內容關鍵字邏輯），並在分支之前插入了 facet-based cap 和 preferred penalty 機制。

**下一步目標**：引入 `ValidationProfile` 資料結構，將每個 `question_type` 分支的邏輯**宣告式化**（declarative），用組態驅動取代 if/elif 硬編碼，為 Phase 4+ 的進一步簡化做準備。

---

## 現有架構（你需要理解的檔案）

### 核心模型（`core/models.py`）

```python
@dataclass(frozen=True)
class FacetSpec:
    required: frozenset[DataFacet]
    preferred: frozenset[DataFacet]

INTENT_FACET_SPECS: dict[Intent, FacetSpec] = {
    Intent.NEWS_DIGEST: FacetSpec(required=frozenset({DataFacet.NEWS}), preferred=frozenset({DataFacet.PRICE_HISTORY})),
    Intent.EARNINGS_REVIEW: FacetSpec(required=frozenset({DataFacet.FINANCIAL_STATEMENTS}), preferred=frozenset({DataFacet.MONTHLY_REVENUE, DataFacet.NEWS})),
    # ... 共 7 個 intent
}
```

### 現有 question_type 清單（共 26 種，17 個有驗證分支）

```python
QUESTION_TYPE_TO_INTENT: dict[str, Intent] = {
    "price_outlook": Intent.VALUATION_CHECK,
    "ex_dividend_performance": Intent.DIVIDEND_ANALYSIS,
    "technical_indicator_review": Intent.TECHNICAL_VIEW,
    "season_line_margin_review": Intent.TECHNICAL_VIEW,
    "fcf_dividend_sustainability_review": Intent.DIVIDEND_ANALYSIS,
    "monthly_revenue_yoy_review": Intent.EARNINGS_REVIEW,
    "shipping_rate_impact_review": Intent.NEWS_DIGEST,
    "electricity_cost_impact_review": Intent.NEWS_DIGEST,
    "macro_yield_sentiment_review": Intent.NEWS_DIGEST,
    "gross_margin_comparison_review": Intent.FINANCIAL_HEALTH,
    "margin_turnaround_review": Intent.EARNINGS_REVIEW,
    "profitability_stability_review": Intent.FINANCIAL_HEALTH,
    "debt_dividend_safety_review": Intent.DIVIDEND_ANALYSIS,
    "pe_valuation_review": Intent.VALUATION_CHECK,
    "fundamental_pe_review": Intent.VALUATION_CHECK,
    "investment_support": Intent.INVESTMENT_ASSESSMENT,
    # ... 另有 10 個沒有驗證分支的 question_type
}
```

### 現有 ValidationLayer 執行流程（`layers/validation_layer.py`）

```
validate()
  1. _calculate_base_confidence() → 五維加權分（evidence 0.30 + trust 0.25 + freshness 0.20 + consistency 0.15 + citation 0.10）
  2. _apply_general_checks() → 全域 warning（bias/empty/preliminary/conflicting/insufficient）
  3. _apply_required_facet_cap() → 三段式 cap（全缺 ≤0.25 / 部分缺 ≤0.50 / 完整不動）
  4. _apply_preferred_facet_penalty() → 每缺一個 preferred facet 扣 0.10，最多扣 0.30
  5. _apply_question_type_rules() → 17 個 if 分支（THIS IS WHAT WE WANT TO REPLACE）
  6. 最終四捨五入 + 映射燈號
```

### 17 個 if 分支的三群分類（Phase 3 已完成）

**Group A（9 個，已改為 warning-only，不再 cap）**：
- `ex_dividend_performance` — 檢查 source_name 含 "dividend" 和 "price"
- `technical_indicator_review` — 檢查 source_name 含 "price"
- `monthly_revenue_yoy_review` — 檢查 source_name 含 "twse" + partial revenue token
- `pe_valuation_review` — 檢查 source_name 含 "twse"
- `profitability_stability_review` — 檢查 financial_statements source + evidence 數量 ≥3 + 特殊推估 warning
- `margin_turnaround_review` — 檢查 financial_statements + "毛利"/"營業利益" 關鍵字 + evidence ≥3
- `fundamental_pe_review` / `investment_support` — 檢查 has_fundamental_evidence + has_valuation_evidence

**Group B（4 個，三段式 cap）**：
- `season_line_margin_review` — 雙信號：price + margin_purchase（全缺 0.25 / 缺一 0.50 / 都有不動）
- `fcf_dividend_sustainability_review` — 雙信號：cashflows + dividend
- `gross_margin_comparison_review` — financial_statements + 雙標的公司名出現在 text
- `debt_dividend_safety_review` — 雙信號：balance_sheet + dividend

**Group C（4 個，保留內容關鍵字邏輯）**：
- `price_outlook` — 複雜三層：數值目標價 / 方向性分析師觀點 / 無依據
- `shipping_rate_impact_review` — 文字關鍵字：紅海/SCFI + 目標價/分析師 + 雙標的
- `electricity_cost_impact_review` — 文字關鍵字：電價成本 + 因應對策 + 雙標的
- `macro_yield_sentiment_review` — 文字關鍵字：CPI/通膨/利率 + 法人觀點

---

## 需要你幫忙設計與實作的東西：ValidationProfile

### 目標

定義一個 `ValidationProfile` dataclass（或 Pydantic model），讓每個 `question_type` 的驗證邏輯可以用**宣告式組態**表達，取代 `_apply_question_type_rules()` 裡的 17 個 if 分支。

### 提議的 ValidationProfile 結構

```python
@dataclass(frozen=True)
class PenaltyRule:
    """單條扣分規則"""
    condition: str          # 條件類型，見下方 enum
    params: dict            # 條件參數
    cap: float | None       # 觸發時的 cap 值（min(score, cap)），None 表示不 cap
    penalty: float | None   # 觸發時的扣分值（score - penalty），None 表示不扣
    warning_template: str   # 觸發時的 warning 訊息模板

@dataclass(frozen=True)
class ValidationProfile:
    """某個 question_type 的驗證組態"""
    required_evidence_keywords: list[str]   # 必要證據關鍵字（source_name 片段）
    required_source_names: list[str]        # 必要來源名稱片段
    penalty_rules: list[PenaltyRule]        # 扣分規則列表（依序評估）
    min_evidence_count: int                 # 最低證據數
```

### 我需要你回答的問題

1. **ValidationProfile 結構是否足夠**表達現有 17 個分支的所有邏輯？逐一檢查每個分支，列出無法被 `required_evidence_keywords` / `required_source_names` / `penalty_rules` / `min_evidence_count` 覆蓋的情境，並提出擴充建議。

2. **PenaltyRule 的 condition 類型**應該有哪些？根據現有分支的邏輯，我至少看到以下 condition 類型：
   - `source_fragment_missing` — source_name 不含某片段
   - `content_keyword_missing` — evidence 合併文字不含某關鍵字
   - `evidence_count_below` — evidence 數量不足
   - `dual_signal_missing` — 兩個信號同時缺失（全缺 vs 缺一）
   - `comparison_company_missing` — 雙標的比較中缺少一方
   - `answer_contains_token` — LLM 回答摘要中出現特定 token
   請補充遺漏的 condition 類型。

3. **與現有 FacetSpec 的關係**：`INTENT_FACET_SPECS` 已經定義了每個 Intent 的 required/preferred facets，而 `_apply_required_facet_cap()` 和 `_apply_preferred_facet_penalty()` 已經在 step 3/4 處理了 facet-level 的 cap。ValidationProfile 的 `required_source_names` 是否會與 FacetSpec 重複？如何劃清職責？

4. **Group C（price_outlook, shipping/electricity/macro）的特殊邏輯**能否也用 ValidationProfile 表達？特別是：
   - `price_outlook` 的三層條件（需呼叫 `is_forward_price_question()`, `is_target_price_question()`, `extract_target_price_values()` 等外部函式）
   - shipping/electricity/macro 的「雙標的公司比較」邏輯
   如果不能，建議如何設計 escape hatch（例如 `custom_validator: Callable | None`）？

5. **VALIDATION_PROFILES 的 registry 設計**：應該放在哪個檔案？建議 `core/validation_profiles.py` 還是直接放在 `core/models.py`？registry 的 key 應該是 `question_type: str` 還是 `(question_type, sub_condition): tuple`（考慮 `price_outlook` 有 forward/directional 兩種子情境）？

6. **遷移策略**：建議分幾個 PR 完成？我傾向：
   - PR 1: 定義 ValidationProfile / PenaltyRule 資料結構 + VALIDATION_PROFILES registry（純新增，不改行為）
   - PR 2: 實作 profile-driven evaluator（`_evaluate_profile()`），先與舊分支並行（dual-run + assert 結果一致）
   - PR 3: 移除舊分支，切換到純 profile-driven
   這個節奏合理嗎？有更好的方式嗎？

7. **測試策略**：現有 `tests/test_validation_layer_phase3.py` 有 7 個 test case。Profile-driven 重構後，需要哪些新測試？建議用 parametrize 對每個 profile 做 exhaustive 測試嗎？

---

## 約束條件

- Python 3.10+，使用 Pydantic v2 和 dataclass
- 必須向下相容：`validate()` 的外部介面不能改變
- 必須通過現有 83+ tests（`pytest tests/ -v`）
- 不能引入新的外部依賴
- 所有現有 cap 值（0.25, 0.50, 0.55, 0.75）必須在重構後產生完全相同的數值結果

---

## 參考檔案路徑

```
src/llm_stock_system/
├── core/
│   ├── models.py              # StructuredQuery, FacetSpec, INTENT_FACET_SPECS, Evidence, GovernanceReport, ValidationResult
│   ├── enums.py               # Intent, DataFacet, ConfidenceLight, TopicTag
│   ├── config.py              # Settings (min_green/yellow_confidence)
│   ├── target_price.py        # is_forward_price_question, extract_target_price_values, etc.
│   └── fundamental_valuation.py # has_fundamental_evidence, has_valuation_evidence
├── layers/
│   └── validation_layer.py    # ValidationLayer class (567 lines, 17 question_type branches)
├── orchestrator/
│   └── pipeline.py            # QueryPipeline, calls validation_layer.validate()
docs/
└── phase3-validation-layer-enhancement.md  # Phase 3 設計文件（三群分類、facet-based cap 設計）
tests/
└── test_validation_layer_phase3.py         # Phase 3 測試（7 cases）
```
