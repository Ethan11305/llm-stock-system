# Phase 3: ValidationLayer Enhancement
## Facet-based Confidence Scoring + 三段式 Cap 遷移

**Status**: Design — Ready for Implementation  
**Estimated Effort**: 3–4 days  
**Depends on**: Phase 0 (FacetSpec / required_facets / preferred_facets), Phase 2 (facet_miss_list from Hydrator)  
**Risk Level**: Medium-High（觸及 14 個現有 question_type 分支，需要精確的回歸覆蓋）

---

## 1. Executive Summary

Phase 2 完成後，`QueryDataHydrator` 已能回報 `facet_miss_list`（哪些 required facets 的 gateway sync 失敗），但 `ValidationLayer` 目前完全忽略這份報告，仍靠 14 個 `question_type` 硬編碼分支決定信心分數的 cap 值。

Phase 3 的目標是：

1. 讓 `ValidationLayer` 讀取 `facet_miss_list` 與 `preferred_miss_list`，以 **facet 存在性** 作為信心分數調整的第一道機制。
2. 把現有的二元 cap（0.25 or 不動）升級為 **三段式 cap**（完全缺失 / 部分缺失 / 完整）。
3. 建立 `question_type` 分支的 **遷移路徑**，分三類處理：可完全替換、可改三段式、保留內容級別檢查。

---

## 2. 現狀分析

### 2.1 資料流現況

```
QueryDataHydrator.hydrate()
    └─ HydrationResult
          ├─ synced_facets: set[DataFacet]      # sync 成功的 facets
          ├─ failed_facets: dict[DataFacet, str] # 所有 sync 失敗的 facets（含 preferred）
          └─ facet_miss_list: list[str]          # 只記錄 required facets 的失敗值

Pipeline.run()
    └─ validation_layer.validate(
           query,
           governance_report,
           answer_draft,
           hydration_result.facet_miss_list   ← 目前只傳這一個
       )

ValidationLayer.validate()
    └─ 收到 facet_miss_list 但完全未使用
    └─ 仍依靠 14 個 query.question_type 分支決定 cap
```

### 2.2 現有 14 個 question_type 分支的問題

目前所有分支都是「二元 cap」：條件不滿足 → `confidence_score = min(score, 0.25)`，沒有中間地帶。

| 分支 | 檢查內容 | 問題 |
|------|----------|------|
| `price_outlook` | 數值目標價 / 方向性分析師觀點 | 內容豐富，但邏輯碎片化 |
| `ex_dividend_performance` | source_name 含 "dividend" 且含 "price" | 可被 facet 完全替換 |
| `technical_indicator_review` | source_name 含 "price" | 可被 facet 完全替換 |
| `season_line_margin_review` | 需同時有 price + margin_purchase source | 可被 facet 三段式替換 |
| `fcf_dividend_sustainability_review` | 需同時有 cashflows + dividend source | 可被 facet 三段式替換 |
| `monthly_revenue_yoy_review` | source_name 含 "twse"（官方月營收） | 可被 facet 替換（+ TWSE 特殊檢查） |
| `shipping_rate_impact_review` | 文字含"紅海"/"SCFI" + 分析師觀點 | 內容關鍵字層面，需保留 |
| `electricity_cost_impact_review` | 文字含電價關鍵字 + 應對措施 | 內容關鍵字層面，需保留 |
| `macro_yield_sentiment_review` | 文字含 CPI / 通膨 + 機構觀點 | 內容關鍵字層面，需保留 |
| `gross_margin_comparison_review` | financial_statements source + 雙標的 | 可被 facet 三段式替換 |
| `margin_turnaround_review` | financial_statements + "毛利率"/"營業利益" | 部分 facet、部分內容 |
| `profitability_stability_review` | financial_statements（多年）| 可被 facet 替換 |
| `debt_dividend_safety_review` | balance_sheet + dividend | 可被 facet 三段式替換 |
| `pe_valuation_review` | source_name 含 "twse"（估值） | 可被 facet 完全替換 |
| `fundamental_pe_review` + `investment_support` | fundamental evidence + valuation evidence | 可被 facet 三段式替換 |

**分類結論**（14 個分支 → 三群）：

- **Group A — 可完全替換為 facet check（9 個）**：`ex_dividend_performance`, `technical_indicator_review`, `monthly_revenue_yoy_review`, `pe_valuation_review`, `profitability_stability_review`, `margin_turnaround_review`, `fundamental_pe_review`/`investment_support`
- **Group B — 可改為三段式 facet + 保留內容 warning（4 個）**：`season_line_margin_review`, `fcf_dividend_sustainability_review`, `gross_margin_comparison_review`, `debt_dividend_safety_review`
- **Group C — 保留內容級別邏輯（3 個）**：`price_outlook`, `shipping_rate_impact_review`, `electricity_cost_impact_review`, `macro_yield_sentiment_review`

---

## 3. 核心設計

### 3.1 缺少的資料：`preferred_miss_list`

**問題**：`facet_miss_list` 只記錄 required facets 的 sync 失敗，但 ValidationLayer 也需要知道 preferred facets 的 sync 情況，才能做扣分。

目前 `HydrationResult` 已有 `failed_facets: dict[DataFacet, str]`（記錄所有失敗的 facets，含 preferred），但 Pipeline 沒有把這個資訊傳給 ValidationLayer。

**解法**：在 `HydrationResult` 加入 `preferred_miss_list: list[str]`，由 Hydrator 填入，Pipeline 一起傳給 ValidationLayer。

```python
# models.py — HydrationResult 擴充
@dataclass
class HydrationResult:
    query_id: str = dataclass_field(default_factory=lambda: str(uuid4()))
    synced_facets: set[DataFacet] = dataclass_field(default_factory=set)
    failed_facets: dict[DataFacet, str] = dataclass_field(default_factory=dict)
    facet_miss_list: list[str] = dataclass_field(default_factory=list)     # required 失敗
    preferred_miss_list: list[str] = dataclass_field(default_factory=list)  # NEW: preferred 失敗
    total_duration_ms: float = 0.0
```

```python
# query_data_hydrator.py — 填入 preferred_miss_list
preferred_facets = set(query.preferred_facets)
# ...在 sync 失敗時：
if facet in required_facets and facet.value not in result.facet_miss_list:
    result.facet_miss_list.append(facet.value)
elif facet in preferred_facets and facet.value not in result.preferred_miss_list:  # NEW
    result.preferred_miss_list.append(facet.value)                                  # NEW
```

```python
# pipeline.py — 也傳 preferred_miss_list
validation_result = self._validation_layer.validate(
    structured_query,
    governance_report,
    answer_draft,
    hydration_result.facet_miss_list,
    hydration_result.preferred_miss_list,  # NEW
)
```

### 3.2 ValidationLayer.validate() 簽名變更

```python
def validate(
    self,
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    facet_miss_list: list[str] | None = None,
    preferred_miss_list: list[str] | None = None,   # NEW (optional, 向下相容)
) -> ValidationResult:
```

### 3.3 三段式 Required Facet Cap 邏輯

在現有五維基礎分計算之後、所有 `question_type` 分支之前，插入一段 facet-based 調整：

```
confidence_score = 五維基礎分 (0.30 + 0.25 + 0.20 + 0.15 + 0.10)

# --- Phase 3 新增：Facet-based cap ---
required_misses = set(facet_miss_list or [])
n_required = len(query.required_facets)
n_missing  = len(required_misses & {f.value for f in query.required_facets})

if n_required > 0:
    if n_missing == n_required:
        # 全部 required facets 都 sync 失敗 → RED cap
        confidence_score = min(confidence_score, 0.25)
        warnings.append(f"All required facets failed to sync: {sorted(required_misses)}")
    elif n_missing > 0:
        # 部分 required facets sync 失敗 → YELLOW cap
        confidence_score = min(confidence_score, 0.50)
        warnings.append(f"Required facet sync failed (partial): {sorted(required_misses)}")
    # 全部 required facets sync 成功 → 不 cap，讓原始分數決定
```

三段式邏輯說明：

| 狀況 | cap | 燈號可能範圍 |
|------|-----|------------|
| required facets 全部 sync 成功 | 不 cap | GREEN / YELLOW / RED（由基礎分決定） |
| 部分 required facets sync 失敗 | ≤ 0.50 | YELLOW 或 RED |
| 全部 required facets sync 失敗 | ≤ 0.25 | RED |

### 3.4 Preferred Facet 扣分邏輯

```
preferred_misses = set(preferred_miss_list or [])
n_preferred_missing = len(preferred_misses & {f.value for f in query.preferred_facets})

if n_preferred_missing > 0:
    preferred_penalty = min(n_preferred_missing * 0.10, 0.30)
    confidence_score = max(confidence_score - preferred_penalty, 0.0)
    warnings.append(
        f"Preferred facets not synced ({n_preferred_missing}): "
        f"{sorted(preferred_misses)}"
    )
```

規則：每缺一個 preferred facet 扣 0.10，最多扣 0.30，確保不因 preferred 問題直接打到紅燈。

### 3.5 `question_type` 分支遷移策略

**原則**：Phase 3 只做「加法」，不刪除任何現有分支。但把每個分支的 cap 行為重新分類：

```
Phase 3 後的執行順序：
1. 五維基礎分計算
2. 全域 warning checks（bias / evidence empty / preliminary / conflicting）
3. Facet-based cap（新增，來自 required_misses / preferred_misses）
4. Question-type 分支（以三類方式繼續存在）
   ├─ Group A：邏輯刪除（保留程式碼但改為 no-op warning，不 cap）
   ├─ Group B：改為三段式 cap（完整→不動，部分→0.50，缺失→0.25）
   └─ Group C：保留原本邏輯（內容關鍵字 + cap，邏輯不變）
```

---

## 4. 各群遷移細節

### 4.1 Group A — 轉為 no-op warning（9 個分支）

這些分支的邏輯在 facet-based scoring 後已被覆蓋。保留分支但移除 `confidence_score = min(...)` 的 cap 行為，改為僅加 warning，避免重複懲罰。

**受影響分支**：

| question_type | 原 cap 條件 | 對應 required facet | Phase 3 後行為 |
|---|---|---|---|
| `ex_dividend_performance` | 缺 dividend 或 price evidence | `DIVIDEND`（required） | 改為 warning only，cap 由 facet-based 決定 |
| `technical_indicator_review` | 缺 price evidence | `PRICE_HISTORY`（required） | 同上 |
| `monthly_revenue_yoy_review` | 缺 twse evidence | `MONTHLY_REVENUE`（preferred） | 同上（preferred penalty 已扣） |
| `pe_valuation_review` | 缺 twse evidence | `PE_VALUATION`（required） | 同上 |
| `profitability_stability_review` | 缺 financial_statements | `FINANCIAL_STATEMENTS`（required） | 同上 |
| `margin_turnaround_review` | 缺 financial_statements | `FINANCIAL_STATEMENTS`（required） | 同上 |
| `fundamental_pe_review` / `investment_support` | 缺 fundamental 或 valuation evidence | `FINANCIAL_STATEMENTS` + `PE_VALUATION`（required） | 同上 |

**修改前（以 `technical_indicator_review` 為例）**：

```python
if query.question_type == "technical_indicator_review":
    source_names = {item.source_name.lower() for item in governance_report.evidence}
    if not any("price" in source_name for source_name in source_names):
        warnings.append("Technical indicator review requires price evidence.")
        confidence_score = min(confidence_score, 0.25)  # ← 刪除這行
```

**修改後**：

```python
if query.question_type == "technical_indicator_review":
    source_names = {item.source_name.lower() for item in governance_report.evidence}
    if not any("price" in source_name for source_name in source_names):
        warnings.append(
            "Technical indicator review: no price evidence in governance report "
            "(facet-based cap already applied if sync failed)."
        )
        # NOTE: cap 已由 facet-based scoring 在步驟 3 處理，此處不再重複 cap
```

> **注意**：這些分支只是「不再 cap」，warning 依然保留。這讓我們保有舊分支的可觀測性，直到 Phase 4 完全移除。

### 4.2 Group B — 改為三段式 cap（4 個分支）

這些分支原本需要「兩個 facets 都存在」才算完整，最適合三段式：

| question_type | Facet A | Facet B | 對應 intent |
|---|---|---|---|
| `season_line_margin_review` | `PRICE_HISTORY`（required） | `MARGIN_DATA`（preferred） | `TECHNICAL_VIEW` |
| `fcf_dividend_sustainability_review` | `DIVIDEND`（required） | `CASH_FLOW`（preferred） | `DIVIDEND_ANALYSIS` |
| `gross_margin_comparison_review` | `FINANCIAL_STATEMENTS`（required） | 雙標的比較（content） | `FINANCIAL_HEALTH` |
| `debt_dividend_safety_review` | `DIVIDEND`（required） | `BALANCE_SHEET`（preferred） | `DIVIDEND_ANALYSIS` |

**修改後（以 `season_line_margin_review` 為例）**：

```python
if query.question_type == "season_line_margin_review":
    source_names = {item.source_name.lower() for item in governance_report.evidence}
    has_price_evidence = any("price" in n for n in source_names)
    has_margin_evidence = any("marginpurchase" in n or "margin_purchase" in n for n in source_names)

    if not has_price_evidence and not has_margin_evidence:
        warnings.append("Season line and margin review: missing both price and margin evidence.")
        confidence_score = min(confidence_score, 0.25)   # 全缺 → RED
    elif not has_price_evidence or not has_margin_evidence:
        warnings.append("Season line and margin review: missing one of price or margin evidence.")
        confidence_score = min(confidence_score, 0.50)   # 部分缺 → YELLOW
    # else: 都有 → 不 cap，但 preferred_miss_list 仍可能已扣了 MARGIN_DATA 的 0.10
```

### 4.3 Group C — 保留現有內容邏輯（3 個分支）

這些分支檢查的是 evidence 的文字內容（關鍵字存在性、數值可信度），不是 facet 存在性，無法被 facet-based scoring 替換。維持現狀，但補上一個 comment 說明不遷移的原因。

**保留分支**：
- `price_outlook`：三層條件（數值目標價 > 方向性分析師觀點 > 無依據），邏輯複雜且正確
- `shipping_rate_impact_review`：需檢查文字中是否真的有"紅海"/"SCFI"等具體信號
- `electricity_cost_impact_review`：需檢查文字中是否有電價與應對措施的具體描述
- `macro_yield_sentiment_review`：需檢查文字中是否有總經指標與機構觀點

這四個分支在 Phase 3 後的角色是「補充性內容品質檢查」，與 facet-based scoring 各司其職：

- Facet-based scoring 回答：「這類問題需要的資料有沒有 sync 進來？」
- Content-based checks 回答：「sync 進來的資料有沒有真正回答問題？」

---

## 5. 修改範圍

### 5.1 需修改的檔案

| 檔案 | 修改內容 | 風險 |
|------|----------|------|
| `core/models.py` | `HydrationResult` 加 `preferred_miss_list` | 低（新欄位，有預設值） |
| `services/query_data_hydrator.py` | 填入 `preferred_miss_list` | 低（對稱於 `facet_miss_list` 邏輯） |
| `orchestrator/pipeline.py` | 傳遞 `preferred_miss_list` 給 ValidationLayer | 低 |
| `layers/validation_layer.py` | 新增 `preferred_miss_list` 參數 + facet-based cap logic + 14 個分支調整 | 高（主要改動） |

### 5.2 不需修改的檔案

- `core/enums.py`：`DataFacet`、`Intent` 已完整（Phase 0）
- `core/models.py` → `INTENT_FACET_SPECS`、`StructuredQuery`：已完整（Phase 0）
- `layers/input_layer.py`：Phase 1 已完整
- `layers/generation_layer.py`：Phase 4 處理

### 5.3 新增測試檔案

- `tests/test_validation_layer_phase3.py`

---

## 6. 測試策略

### 6.1 新增測試（`test_validation_layer_phase3.py`）

#### Case 1：所有 required facets 成功 sync
```python
# facet_miss_list = []
# 期待：facet-based scoring 不影響 confidence_score
# 燈號完全由基礎分決定
```

#### Case 2：部分 required facets 失敗
```python
# intent = INVESTMENT_ASSESSMENT
# required_facets = {FINANCIAL_STATEMENTS, PE_VALUATION}
# facet_miss_list = ["pe_valuation"]  # 一個 required 失敗
# 期待：confidence_score <= 0.50, light = YELLOW
# 期待：warnings 包含 "Required facet sync failed (partial)"
```

#### Case 3：全部 required facets 失敗
```python
# intent = EARNINGS_REVIEW
# required_facets = {FINANCIAL_STATEMENTS}
# facet_miss_list = ["financial_statements"]
# 期待：confidence_score <= 0.25, light = RED
# 期待：warnings 包含 "All required facets failed to sync"
```

#### Case 4：preferred facets 失敗（扣分）
```python
# intent = TECHNICAL_VIEW
# required_facets = {PRICE_HISTORY}  # sync 成功
# preferred_facets = {MARGIN_DATA}
# preferred_miss_list = ["margin_data"]
# 期待：confidence_score 比基礎分低 0.10
# 期待：warnings 包含 "Preferred facets not synced"
```

#### Case 5：preferred 多個失敗（扣分上限 0.30）
```python
# intent = DIVIDEND_ANALYSIS
# preferred_miss_list = ["cash_flow", "balance_sheet", "financial_statements"]
# 期待：扣分不超過 0.30
```

#### Case 6：Group B 三段式 cap 驗證
```python
# question_type = "season_line_margin_review"
# 情境 A：price 有，margin 沒有 → cap 0.50
# 情境 B：兩個都沒有 → cap 0.25
# 情境 C：兩個都有 → 不 cap
```

#### Case 7：向下相容（不傳 preferred_miss_list）
```python
# validate(query, governance_report, answer_draft, facet_miss_list=None)
# 期待：不 crash，行為與 Phase 2 相同
```

### 6.2 回歸測試

確認以下測試全部通過（Phase 2 後共 83+ passed）：

```bash
pytest tests/ -v --tb=short
```

重點關注：
- `test_validation_layer_phase3.py`（新）
- `test_query_data_hydrator_phase2.py`（確認 HydrationResult 結構未破壞）
- `test_pipeline.py`（確認 pipeline 資料流正確）
- 所有 `test_*_queries.py`（確認各 question_type 行為未意外改變）

---

## 7. 實作執行順序

```
Step 1: core/models.py
    → HydrationResult 加 preferred_miss_list 欄位

Step 2: services/query_data_hydrator.py
    → 在 sync 失敗時填入 preferred_miss_list

Step 3: orchestrator/pipeline.py
    → 傳遞 preferred_miss_list 給 validate()

Step 4: layers/validation_layer.py — 分三個子步驟
    Step 4a: 修改函式簽名，加 preferred_miss_list 參數
    Step 4b: 在現有警告 checks 之後、question_type 分支之前，插入 facet-based cap 邏輯
    Step 4c: 逐一調整 14 個 question_type 分支（Group A: 移除 cap / Group B: 三段式 / Group C: 不動）

Step 5: tests/test_validation_layer_phase3.py
    → 依照第 6.1 節的 case 逐一新增

Step 6: pytest 全回歸
    → 確認 83+ tests 全過
```

---

## 8. 邊界情況與注意事項

### 8.1 `facet_miss_list` 的語義限制

`facet_miss_list` 追蹤的是 **gateway sync 失敗**，不是 **evidence 為空**。即使 sync 成功，governance report 的 evidence 可能仍然沒有對應類型的文件（例如資料庫裡本來就沒有這支股票的財報）。

這意味著 facet-based cap 是「樂觀的」：sync 成功 → facet-based 不 cap，但 Group C 的內容關鍵字檢查仍然可能 cap。

這個設計是有意的：facet-based scoring 負責「基礎設施層面」的信心，content-based checks 負責「資料品質層面」的信心，兩者互補。

### 8.2 雙重 cap 的疊加

當 facet-based cap 和 question_type content cap 同時觸發，`min()` 的連鎖效果正確：

```python
# 例：required facet 部分缺失 → cap 0.50
# 且 content check 也失敗 → cap 0.25
# 結果：min(min(score, 0.50), 0.25) = min(score, 0.25) → RED，正確
```

Group A 分支移除 cap 後，這個疊加效果只由 facet-based + Group C 產生，不會有「facet cap 已 0.25，Group A 再 cap 0.25」的重複懲罰。

### 8.3 preferred_miss_list 為 None 的情況

當 `QueryHydrator` 未執行（`query_hydrator` is None in pipeline），`HydrationResult()` 預設值為空 list，`preferred_miss_list=None` 也不會觸發任何扣分，行為與 Phase 2 完全相同。

### 8.4 `monthly_revenue_yoy_review` 的 TWSE 特殊情況

這個分支有兩條 cap：
1. 無 TWSE source → 0.25（可被 MONTHLY_REVENUE facet 替換）
2. "僅更新到今年前" 在 summary 中 → 0.75（內容級別，應保留）

Phase 3 只移除第一條 cap（改為 warning only），第二條保留。

---

## 9. Definition of Done

進入 Phase 4 之前，以下條件必須全部成立：

- [ ] `HydrationResult` 有 `preferred_miss_list` 欄位
- [ ] `QueryDataHydrator` 正確填入 `preferred_miss_list`
- [ ] `pipeline.py` 傳遞 `preferred_miss_list` 給 `ValidationLayer`
- [ ] `ValidationLayer.validate()` 接受 `preferred_miss_list` 參數（optional，向下相容）
- [ ] Facet-based 三段式 cap 邏輯在 question_type 分支之前執行
- [ ] Group A（9 個分支）的 `confidence_score = min(...)` cap 已移除，僅保留 warning
- [ ] Group B（4 個分支）已改為三段式 cap（完整→不動，部分→0.50，缺失→0.25）
- [ ] Group C（4 個分支）邏輯不變
- [ ] `tests/test_validation_layer_phase3.py` 的所有 case 通過
- [ ] 全部回歸測試通過（pytest tests/ 無 failure）

---

## 10. 後續 Phase 預告

Phase 3 完成後，ValidationLayer 仍保留 14 個 question_type 分支的程式碼（Group A 已 no-op，Group B 已三段式，Group C 不動）。Phase 4 的任務是：

- **Phase 4**：修改 GenerationLayer / LLM prompt 的 question_type 分支，改為 Intent + topic_tags 模板驅動。ValidationLayer 的 Group A no-op 分支在 Phase 4 後可以完全移除。
- **Phase 5**：LLM-driven generation 全面接管，ValidationLayer 的 Group C content checks 可能需要重新評估是否仍有意義。
