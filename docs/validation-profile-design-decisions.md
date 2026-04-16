# ValidationProfile 設計決策書

**Status**: Design Finalized — Ready for PR1  
**Date**: 2026-04-16  
**Based on**: Codex review of `codex-prompt-validation-profile.md`

---

## 1. 七大設計決策

### 決策 1：用 dataclass，不用 Pydantic

ValidationProfile / ValidationRule 是程式內部靜態規則，不是外部輸入，不需要 Pydantic 的序列化/驗證能力。用 `@dataclass(frozen=True)` 保持輕量。

### 決策 2：PenaltyRule → ValidationRule，收斂頂層欄位

重新命名為 `ValidationRule`，因為規則不只是 penalty，也包含 warning-only 和 cap-only。

ValidationProfile 收斂為四個核心欄位：

```python
@dataclass(frozen=True)
class ValidationProfile:
    question_type: str
    min_evidence_count: int
    rules: list[ValidationRule]
    custom_validator: CustomValidatorFn | None = None
```

原始提案的 `required_evidence_keywords` 和 `required_source_names` **不做頂層欄位**，改收進 `ValidationRule.params` 裡，避免語義重疊。

### 決策 3：condition 用 Enum，不用自由字串

```python
class ConditionKind(str, Enum):
    SOURCE_FRAGMENT_MISSING = "source_fragment_missing"
    CONTENT_KEYWORD_MISSING = "content_keyword_missing"
    EVIDENCE_COUNT_BELOW = "evidence_count_below"
    DUAL_SIGNAL_MISSING = "dual_signal_missing"
    COMPARISON_COMPANY_MISSING = "comparison_company_missing"
    ANSWER_CONTAINS_TOKEN = "answer_contains_token"
```

超過這六種的複雜度 → 直接走 `custom_validator`，不擴 DSL。

### 決策 4：FacetSpec 與 ValidationProfile 明確分層

| 層 | 職責 | 位置 | 時機 |
|---|---|---|---|
| **FacetSpec** (INTENT_FACET_SPECS) | 資料取得契約 | `_apply_required_facet_cap()` + `_apply_preferred_facet_penalty()` (step 3/4) | question_type 分支之前 |
| **ValidationProfile** | 證據語義檢查 | `_apply_question_type_rules()` (step 5) | facet cap 之後 |

ValidationProfile **不重複宣告 facet truth**，不把 required_source_names 當第一級規則來源。兩層互補不重疊：
- FacetSpec 回答：「需要的資料有沒有 sync 進來？」
- ValidationProfile 回答：「sync 進來的資料有沒有真正回答問題？」

### 決策 5：Group C 用 typed custom_validator

```python
CustomValidatorFn = Callable[
    [StructuredQuery, GovernanceReport, AnswerDraft, float, list[str]],
    float
]
```

- `price_outlook`：**一定要用** custom_validator（依賴 `target_price.py` 的 `is_forward_price_question()` / `is_target_price_question()` / `extract_target_price_values()` 等 query-aware 函式）
- `shipping_rate_impact_review` / `electricity_cost_impact_review` / `macro_yield_sentiment_review`：**先用 declarative rules**，如果 parity test 發現 keyword 噪音太高再升級

### 決策 6：Registry 放 `core/validation_profiles.py`

- 不塞進 `core/models.py`（已有 QUESTION_TYPE_TO_INTENT + INTENT_FACET_SPECS，太肥）
- Key 維持 `question_type: str`，不用 tuple
- `price_outlook` 的 forward/directional 分流由 profile 內的 custom_validator 自己判斷

```python
# core/validation_profiles.py
VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    "ex_dividend_performance": ValidationProfile(...),
    "technical_indicator_review": ValidationProfile(...),
    # ... 共 17 個
}
```

### 決策 7：PR 切法 — 3 個 PR

| PR | 內容 | 風險 |
|---|---|---|
| **PR1** | 新增 `ValidationRule` / `ValidationProfile` / `ConditionKind` / `VALIDATION_PROFILES` registry / `_evaluate_profile()` evaluator。**完全不改既有行為**，新舊並存。 | 低（純新增） |
| **PR2** | 搬 Group A（9 個 warning-only）+ Group B（4 個三段式 cap）到 profile-driven。做 dual-run parity tests，production 可 fallback legacy。 | 中 |
| **PR3** | 搬 Group C（4 個含 price_outlook），確認 parity 後切換 profile-first，最後刪除 legacy 分支。 | 高 |

---

## 2. 資料結構定義（PR1 的精確規格）

### 2.1 ConditionKind Enum

```python
# core/validation_profiles.py

from enum import Enum

class ConditionKind(str, Enum):
    """ValidationRule 支援的條件類型"""
    SOURCE_FRAGMENT_MISSING = "source_fragment_missing"
    # params: {"fragments": ["price", "dividend"]}
    # 語義：source_names 中沒有任何一個含 fragments 中的任一片段

    CONTENT_KEYWORD_MISSING = "content_keyword_missing"
    # params: {"keywords": ["紅海", "scfi", "運價"], "match_mode": "any"|"all"}
    # 語義：evidence 合併文字(normalized)不含 keywords

    EVIDENCE_COUNT_BELOW = "evidence_count_below"
    # params: {"threshold": 3}
    # 語義：evidence 數量 < threshold

    DUAL_SIGNAL_MISSING = "dual_signal_missing"
    # params: {
    #   "signal_a_fragments": ["price"],
    #   "signal_b_fragments": ["marginpurchase", "margin_purchase"],
    #   "both_missing_cap": 0.25,
    #   "one_missing_cap": 0.50
    # }
    # 語義：雙信號檢查（全缺 / 缺一 / 都有）

    COMPARISON_COMPANY_MISSING = "comparison_company_missing"
    # params: {} (從 query.comparison_ticker 自動推導)
    # 語義：雙標的比較中，evidence text 缺少一方公司名

    ANSWER_CONTAINS_TOKEN = "answer_contains_token"
    # params: {"tokens": ["部分月份", "僅能確認", "資料僅到"]}
    # 語義：LLM answer_draft.summary 含特定 token
```

### 2.2 ValidationRule

```python
@dataclass(frozen=True)
class ValidationRule:
    """單條驗證規則"""
    condition: ConditionKind
    params: dict[str, Any]
    cap: float | None = None        # 觸發時 min(score, cap)
    penalty: float | None = None    # 觸發時 score -= penalty
    warning: str = ""               # 觸發時追加的 warning 訊息
```

### 2.3 ValidationProfile

```python
from typing import Callable, Any

CustomValidatorFn = Callable[
    [
        "StructuredQuery",
        "GovernanceReport",
        "AnswerDraft",
        float,          # current confidence_score
        list[str],      # warnings (mutable, append to it)
    ],
    float,  # return adjusted confidence_score
]

@dataclass(frozen=True)
class ValidationProfile:
    """某個 question_type 的驗證組態"""
    question_type: str
    min_evidence_count: int = 0
    rules: tuple[ValidationRule, ...] = ()
    custom_validator: CustomValidatorFn | None = None
```

### 2.4 VALIDATION_PROFILES 範例（部分）

```python
VALIDATION_PROFILES: dict[str, ValidationProfile] = {

    # --- Group A: warning-only ---
    "ex_dividend_performance": ValidationProfile(
        question_type="ex_dividend_performance",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={"fragments": ["dividend"]},
                warning="Ex-dividend performance: missing dividend evidence (facet-based cap already applied if sync failed).",
            ),
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={"fragments": ["price"]},
                warning="Ex-dividend performance: missing price evidence (facet-based cap already applied if sync failed).",
            ),
        ),
    ),

    # --- Group B: 三段式 cap ---
    "season_line_margin_review": ValidationProfile(
        question_type="season_line_margin_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.DUAL_SIGNAL_MISSING,
                params={
                    "signal_a_fragments": ["price"],
                    "signal_b_fragments": ["marginpurchase", "margin_purchase"],
                    "both_missing_cap": 0.25,
                    "one_missing_cap": 0.50,
                },
                warning="Season line and margin review: missing {detail}.",
            ),
        ),
    ),

    # --- Group C: custom validator ---
    "price_outlook": ValidationProfile(
        question_type="price_outlook",
        custom_validator=_validate_price_outlook,  # 指向獨立函式
    ),
}
```

---

## 3. Evaluator 邏輯（PR1 新增，與舊分支並行）

```python
def _evaluate_profile(
    self,
    profile: ValidationProfile,
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    source_names = self._source_names(governance_report)
    combined_text = self._combined_text(governance_report)
    normalized_text = combined_text.lower()

    # 1. min_evidence_count 檢查
    if profile.min_evidence_count > 0 and len(governance_report.evidence) < profile.min_evidence_count:
        warnings.append(f"Evidence count ({len(governance_report.evidence)}) below minimum ({profile.min_evidence_count}).")

    # 2. 依序評估 rules
    for rule in profile.rules:
        triggered = self._check_condition(rule, query, governance_report, answer_draft, source_names, normalized_text)
        if triggered:
            if rule.warning:
                warnings.append(rule.warning)
            if rule.cap is not None:
                confidence_score = min(confidence_score, rule.cap)
            if rule.penalty is not None:
                confidence_score = max(confidence_score - rule.penalty, 0.0)

    # 3. custom_validator (escape hatch)
    if profile.custom_validator is not None:
        confidence_score = profile.custom_validator(query, governance_report, answer_draft, confidence_score, warnings)

    return confidence_score
```

---

## 4. 測試策略

### 4.1 Rule Engine 單元測試（新增）

Parametrize 每種 `ConditionKind` 的語義：

```python
@pytest.mark.parametrize("condition,params,expected", [
    (ConditionKind.SOURCE_FRAGMENT_MISSING, {"fragments": ["price"]}, True),   # 無 price source
    (ConditionKind.SOURCE_FRAGMENT_MISSING, {"fragments": ["price"]}, False),  # 有 price source
    (ConditionKind.CONTENT_KEYWORD_MISSING, {"keywords": ["紅海"], "match_mode": "any"}, True),
    (ConditionKind.EVIDENCE_COUNT_BELOW, {"threshold": 3}, True),  # evidence=2
    # ...
])
def test_condition_evaluation(condition, params, expected):
    ...
```

### 4.2 Profile Behavior 測試（新增）

Parametrize question_type + evidence set + expected cap + warning substring：

```python
@pytest.mark.parametrize("question_type,evidence_sources,expected_cap,warning_substr", [
    # Group A: warning-only
    ("ex_dividend_performance", ["dividend", "price"], None, None),           # happy
    ("ex_dividend_performance", ["dividend"], None, "missing price"),         # degraded (warning only)

    # Group B: 三段式
    ("season_line_margin_review", ["price", "marginpurchase"], None, None),   # happy
    ("season_line_margin_review", ["price"], 0.50, "missing one"),            # 缺一
    ("season_line_margin_review", [], 0.25, "missing both"),                  # 全缺

    # Group C: custom
    ("price_outlook", [...], 0.55, "scenario-dependent"),  # numeric target
    ("price_outlook", [...], 0.55, "directional"),         # directional only
    ("price_outlook", [...], 0.25, "lacks"),               # no context
])
def test_profile_behavior(question_type, evidence_sources, expected_cap, warning_substr):
    ...
```

### 4.3 Parity 測試（PR2/PR3）

對每個 question_type 做 dual-run：`_apply_question_type_rules()` vs `_evaluate_profile()`，assert 兩者回傳的 `confidence_score` 完全一致。

---

## 5. 執行順序

```
PR1（純新增，不改行為）
├── core/validation_profiles.py
│   ├── ConditionKind enum
│   ├── ValidationRule dataclass
│   ├── ValidationProfile dataclass
│   ├── CustomValidatorFn type alias
│   └── VALIDATION_PROFILES registry（17 個 profile 全部定義）
├── layers/validation_layer.py
│   └── _evaluate_profile() 方法（新增，不呼叫）
└── tests/test_validation_rule_engine.py
    └── ConditionKind 單元測試

PR2（Group A + B 遷移）
├── layers/validation_layer.py
│   └── _apply_question_type_rules() 內，Group A/B 改為呼叫 _evaluate_profile()
├── tests/test_validation_profile_parity.py
│   └── 13 個 profile 的 dual-run parity tests
└── pytest 全回歸

PR3（Group C 遷移 + legacy 清除）
├── core/validation_profiles.py
│   └── _validate_price_outlook() / shipping / electricity / macro custom validators
├── layers/validation_layer.py
│   └── _apply_question_type_rules() 改為純 profile-driven，刪除所有 legacy 分支
├── tests/test_validation_profile_parity.py
│   └── Group C 的 parity tests + price_outlook 三態測試
└── pytest 全回歸
```

---

## 6. 風險與注意事項

1. **DUAL_SIGNAL_MISSING 的 warning 模板**：現有分支的 warning 訊息在「全缺」和「缺一」時不同，ValidationRule 的 `warning` 欄位需要支援 `{detail}` 佔位符，由 evaluator 在運行時替換。
2. **profitability_stability_review 的特殊 cap**：它有一條「若只看財報結構推估 + 無 news source → cap 0.75」的規則，這是 Group A 中唯一仍有 cap 行為的分支。需要確認這條是走 ValidationRule 還是 custom_validator。
3. **monthly_revenue_yoy_review 的雙重規則**：Phase 3 移除了第一條 cap（TWSE source），但保留了第二條（partial revenue token in summary → warning only）。Profile 裡需要兩條 rule。
4. **gross_margin_comparison_review 混合邏輯**：同時需要 source_fragment check（financial_statements）和 comparison_company check 和 evidence_count check，三者組合可能需要特別注意 rule 評估順序。
