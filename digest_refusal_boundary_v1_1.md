# Digest Refusal Boundary 規格書 v1.1 (Final)

> Status: Implementation-ready
> Scope: `single_stock_digest` 產品線
> 適用對象: `digest/` 路徑；不適用於 `legacy_qa/`

---

## 1. 文件目的

本文件定義 `single_stock_digest` 產品線的 Refusal Boundary，用以回答以下核心問題：

1. 什麼情況下系統應直接拒絕生成 digest？
2. 什麼情況下系統可生成降級版 digest？
3. 什麼情況下系統可正常生成 digest？
4. 上述判斷應如何被記錄、回查、分析與校準？

本文件的目標不是讓系統「盡量回答」，而是讓系統在資訊不足、分類不明、來源不可信或範圍不符時，明確拒絕，避免輸出低品質但外觀完整的摘要。

---

## 2. 適用範圍

本規格只適用於 digest 產品線：

- `queryProfile = SINGLE_STOCK_DIGEST`
- 單一股票
- 預設近 7 天
- 主題限於：
  - `NEWS`
  - `ANNOUNCEMENT`
  - `COMPOSITE`

本規格不適用於 legacy QA 路徑，也不適用於多題型股票問答的通用流程。

---

## 3. 產品定位

Digest 產品線的核心定位是：

> 對單一股票近 7 天內的新聞 / 公告進行可信摘要，並在資料不足時拒絕生成，而非勉強給出低信心答案。

因此，digest 與 legacy QA 的最大差異不是「摘要形式」而是：

- digest 有封閉 scope
- digest 有 refusal boundary
- digest 有 query log 可追溯契約
- digest 不依賴 `question_type -> intent` 推導
- digest 不允許無限制擴張的 fallback tags

---

## 4. 設計原則

### 4.1 Refusal 先於 Validation
先判斷「能不能答」，再判斷「答得有多好」。

### 4.2 拒絕優先於猜測
資訊不足、範圍不符、分類不明時，應拒絕回答，而不是用模糊補全硬湊 digest。

### 4.3 Digest 不依賴 `question_type`
digest 路徑不得使用：

- `infer_intent_from_question_type()`
- `QUESTION_TYPE_FALLBACK_TOPIC_TAGS`
- 任意形式的 `question_type -> intent` 反推

### 4.4 Scope 必須封閉
digest 只服務定義明確的請求；所有超出範圍的查詢都應及早退出。

### 4.5 可觀測性優先
所有 refusal / degraded / normal 決策都必須可被 query log 回查、切片、統計、校準。

---

## 5. 術語定義

本節為所有後續規則的基礎定義。任何規則中使用的函式或概念，必須在本節已定義。

### 5.1 高可信來源（High-Trust Source）

`high_trust_source` 指 `Evidence.source_tier == HIGH` 的證據來源。

本規格不在此重新發明來源分級邏輯，而是引用 shared source-tier policy。v1.1 對照如下：

**HIGH**
- 公司公告 / 公開資訊觀測站 / 交易所官方資料
- 主管機關、交易所官方 API / 官方網站
- 公司法說會逐字稿、官方新聞稿

**MEDIUM**
- 主流財經媒體
- 有編輯制度的新聞媒體
- 大型資料供應商整理頁

**LOW**
- 個人部落格
- 社群貼文
- 未驗證轉載站
- 無明確編輯責任之內容來源

Refusal 與 degraded 判斷只依 `source_tier` 運作，不直接以媒體名稱寫死在此規格。

### 5.2 交叉驗證（Cross-Validation）

`cross_validated(evidence)` 的 v1.1 定義：

> 在同一個 digest query 中，至少存在 2 筆**不同 URL 且來源組織不同**的 evidence，支撐同一個核心事件或同一事實群組。

**不算交叉驗證**

- 同一發布者的兩篇改寫稿
- 同一篇文章的鏡像轉載
- 只有主題接近，但不是同一事件 / 同一事實

**v1.1 實作建議**

初版可用：

- `normalized_event_key`
- 或 title / excerpt 相似分群
- 再加 `source_name` 去重

### 5.3 Topic Mismatch

`topic_mismatch(query, evidence)` 的 v1.1 定義：

> 當 digest query 指定 topic 後，evidence 中符合該 topic 的比例**低於 50%**，視為 `TOPIC_MISMATCH`。

**Topic 對應**

- `NEWS`：evidence 主要應來自新聞類文件
- `ANNOUNCEMENT`：evidence 主要應來自公告 / official disclosure 類文件
- `COMPOSITE`：兩者皆可（此 topic 下 topic_mismatch 永遠為 False）

**v1.1 檢查方式**

topic mismatch 應優先用 document metadata / source_type / topic label 檢查，**不重新跑 classifier**。

### 5.4 近 7 天

`within_7d(published_at, query_time)` 的 v1.1 定義：

- 時區：`Asia/Taipei`
- 規則：`published_at >= query_time - 7 * 24h`

這裡使用 calendar time，不是交易日，也不是日曆天 00:00 切點。

### 5.5 Digest 封閉集合

digest path 的 v1.1 封閉集合如下。

**Allowed Intent**

- `SINGLE_STOCK_DIGEST`

**Allowed Topic**

- `NEWS`
- `ANNOUNCEMENT`
- `COMPOSITE`

**Forbidden Scope**

- 多股票比較
- 技術分析
- 估值 / 合理價 / 本益比 / 目標價
- forecast / price outlook / 漲跌預測
- earnings deep dive
- 使用者明確指定非 7 天範圍

### 5.6 Rule Fallback

digest path 允許非常有限的 rule fallback，但只能補「保守 metadata」，不能補「核心語意」。

**合法的 Rule Fallback**

- `topic` 的保守判斷
- `timeRangeExplicit = False` 時套預設 `7d`
- 使用封閉關鍵字表補少量高精確 `TopicTag`（見 §6.3）

**非法的 Rule Fallback**

- `question_type -> intent`
- `question_type -> fallback tags`
- 對模糊 query 進行推測性意圖補全

### 5.7 freshness_strong

`freshness_strong(evidence, query_time)` 的 v1.1 定義。

對於 `single_stock_digest`，若要視為 strong freshness，必須同時滿足：

1. **所有 evidence 都在近 7 天內**
   - 對每筆 `item`：`within_7d(item.published_at, query_time) == True`

2. **依 topic 的 recency 條件**：
   - 若 `topic == ANNOUNCEMENT`：
     - 至少 1 筆 `HIGH` source evidence 在近 72 小時內
   - 否則 (`NEWS` / `COMPOSITE`)：
     - 至少 1 筆 evidence 在近 48 小時內

**v1.1 實作建議**

```python
def freshness_strong(evidence: list[Evidence], query_time: datetime) -> bool:
    if not all(within_7d(e.published_at, query_time) for e in evidence):
        return False

    if topic_of(evidence) == Topic.ANNOUNCEMENT:
        return any(
            e.source_tier == SourceTier.HIGH
            and e.published_at >= query_time - timedelta(hours=72)
            for e in evidence
        )

    # NEWS / COMPOSITE
    return any(
        e.published_at >= query_time - timedelta(hours=48)
        for e in evidence
    )
```

### 5.8 evidence_conflict

`evidence_conflict(governance_report)` 的 v1.1 定義。

**v1.1 實作**：固定回傳 `False`。

**v1.1 立場**：evidence 間的語意衝突偵測需要 governance layer 具備結構化事實抽取能力。此能力不在 v1.1 digest 範圍內。`D4 EVIDENCE_CONFLICT` enum 保留於 schema 與 log，但在 v1.1 不會被觸發。

**v1.2+ 升級路徑**

- governance 層加入 `ExtractedFact` 抽取
- `evidence_conflict` 改為依 `extracted_facts` 的規則比較
- 改動不需要修改 refusal spec 契約，也不需要修改 query log schema 或 UI 契約

**v1.1 實作**

```python
def evidence_conflict(governance_report: GovernanceReport) -> bool:
    # v1.1: dormant, see §5.8 and §18.5
    return False
```

### 5.9 classifier_tag_coverage

`classifier_tag_coverage` 的 v1.1 定義。

**來源**

`classifier_tag_coverage` 由 digest input / classification layer 的 **deterministic post-processing** 計算，**不是**由 classifier LLM 直接最終決定。

**v1.1 流程**

1. classifier LLM 輸出：
   - `predicted_topic`
   - `predicted_tags`
   - `classifier_raw_score`

2. digest 規則層根據 `topic` 查出對應的 `required_tags`（見 §5.9.1）

3. 用 `predicted_tags ∩ required_tags` 的命中比例計算 `classifier_tag_coverage`

**值域**

```
{"sufficient", "partial", "insufficient"}
```

**v1.1 計算規則**

設：

- `required_tags` = 該 topic 對應的 minimum tag set
- `matched_tags` = `predicted_tags ∩ required_tags`
- `coverage_ratio` = `len(matched_tags) / len(required_tags)`

則：

- `sufficient`：`coverage_ratio == 1.0`
- `partial`：`coverage_ratio >= 0.5` 且 `< 1.0`
- `insufficient`：`coverage_ratio < 0.5`

若 `required_tags` 為空集合，直接回傳 `"sufficient"`。

**v1.1 實作建議**

```python
def compute_classifier_tag_coverage(
    topic: Topic,
    predicted_tags: set[TopicTag],
) -> str:
    required_tags = MINIMUM_TAG_SET_BY_TOPIC[topic]
    if not required_tags:
        return "sufficient"

    matched_tags = predicted_tags & required_tags
    coverage_ratio = len(matched_tags) / len(required_tags)

    if coverage_ratio == 1.0:
        return "sufficient"
    if coverage_ratio >= 0.5:
        return "partial"
    return "insufficient"
```

### 5.9.1 MINIMUM_TAG_SET_BY_TOPIC

digest path 的封閉表。

**v1.1 值**

```python
MINIMUM_TAG_SET_BY_TOPIC: dict[Topic, set[TopicTag]] = {
    Topic.NEWS: set(),
    Topic.ANNOUNCEMENT: set(),
    Topic.COMPOSITE: set(),
}
```

**v1.1 立場**

v1.1 將此表所有值設為空集合。此選擇對應 §18.5 Dormant 規則：R9 `INSUFFICIENT_TAG_COVERAGE` 與 D5 `PARTIAL_TAG_COVERAGE` 在 v1.1 不觸發。

**硬限制**

- `sum(len(v) for v in MINIMUM_TAG_SET_BY_TOPIC.values()) <= 5`
- CI 應加測試強制此上限
- 任一新增需 spec PR review
- 不允許多層 fallback map
- 不允許把 `question_type` 重新接回 digest path

**v1.2+ 升級路徑**

當 R9 / D5 啟用時（見 §18.6），再以資料驅動方式填入 required tags。**不要在無 query log 證據的情況下猜測 required set。**

---

## 6. Digest 封閉輸入與 Rule-Based Scope Guard

### 6.1 DigestQuery 最低必要欄位

digest 路徑的輸入模型至少需具備：

- `user_query`
- `ticker`
- `company_name`
- `topic`
- `time_range_label`
- `time_range_days`
- `time_range_explicit: bool`
- `intent = SINGLE_STOCK_DIGEST`
- `controlled_tags`
- `topic_tags`
- `classifier_source`
- `classifier_raw_score`
- `classifier_tag_coverage`
- `query_profile = SINGLE_STOCK_DIGEST`

### 6.2 `R4b` 的 v1.1 偵測方式

`OUT_OF_SCOPE_QUERY_KIND` 的 v1.1 偵測**不依賴 classifier**，而採用封閉式 rule-based scope guard。

**建議實作位置**

`digest/input/scope_guard.py`

**v1.1 關鍵字清單**

```python
OUT_OF_SCOPE_KEYWORDS: dict[str, str] = {
    "估值": "VALUATION",
    "本益比": "VALUATION",
    "合理價": "VALUATION",
    "目標價": "PRICE_TARGET",
    "技術線": "TECHNICAL",
    "K線": "TECHNICAL",
    "均線": "TECHNICAL",
    "RSI": "TECHNICAL",
    "MACD": "TECHNICAL",
    "預測": "FORECAST",
    "會漲嗎": "FORECAST",
    "股價會到": "FORECAST",
    "EPS": "EARNINGS_DEEP_DIVE",
    "毛利率": "EARNINGS_DEEP_DIVE",
    "現金流": "EARNINGS_DEEP_DIVE",
    "vs": "MULTI_STOCK",
    "比較": "MULTI_STOCK",
    "哪個比較好": "MULTI_STOCK",
}
```

**規則**

- v1.1 為封閉清單
- 長度建議不超過 30
- 任一新增都需 PR review
- scope guard 在 classifier 前或與 classifier 並行皆可，但必須**先於 retrieval refusal**

### 6.3 `ALLOWED_KEYWORD_TAGS` 的封閉清單

**建議實作位置**

`digest/input/rule_fallback.py`

```python
ALLOWED_KEYWORD_TAGS: dict[str, TopicTag] = {
    "法說": TopicTag.GUIDANCE,
    "指引": TopicTag.GUIDANCE,
    "AI": TopicTag.AI,
    "電動車": TopicTag.EV,
    "航運": TopicTag.SHIPPING,
    "電價": TopicTag.ELECTRICITY,
    "殖利率": TopicTag.MACRO,
}
```

**硬限制**

- `len(ALLOWED_KEYWORD_TAGS) <= 10`
- CI 應加測試強制此上限
- 不允許多層 fallback map
- 不允許把 `question_type` 重新接回 digest path

---

## 7. 三態結果模型

digest query 的結果只有三種：

- `REFUSE`
- `DEGRADED`
- `NORMAL`

### 7.1 REFUSE

系統拒絕生成 digest。

**規則**

- `confidenceScore = null`
- `confidenceLight = null`
- 必須帶：
  - `refusalCategory`
  - `refusalReason`

**REFUSE 不屬於 confidence 體系。**

### 7.2 DEGRADED

系統允許輸出降級版 digest，但需顯式標示品質限制。

**規則**

- `confidenceScore ∈ [0.30, 0.69]`
- `confidenceLight ∈ {RED, YELLOW}`
- `degradedReasons` 為 list，可同時多個

### 7.3 NORMAL

系統允許輸出完整 digest。

**規則**

- `confidenceScore ∈ [0.70, 1.00]`
- `confidenceLight = GREEN`

v1.1 起，NORMAL 不再允許 YELLOW，避免 UI 歧義。

---

## 8. 三階段 Early-Exit 流程

refusal 不是單一 gate，而是三個 checkpoint。

**Checkpoint 1：Early Refusal**
位置：parse / scope guard / classifier 後，retrieval 前

**Checkpoint 2：Retrieval Refusal**
位置：retrieval 後，governance 前

**Checkpoint 3：Governance Decision**
位置：governance 後，presentation 前

---

## 9. Early Refusal 規則

### R1 `UNRESOLVED_TICKER`

**條件**

- `ticker is None`

**動作**

- `REFUSE(PARSE, UNRESOLVED_TICKER)`

### R2 `CLASSIFIER_UNKNOWN`

**條件**

- classifier 無法確認為 digest query
- classifier raw score 低於最低接受閾值
- classifier 回傳 `unknown`

**動作**

- `REFUSE(CLASSIFICATION, CLASSIFIER_UNKNOWN)`

### R4 `EXPLICIT_OUT_OF_SCOPE_TIME_RANGE`

**條件**

- `time_range_explicit == True`
- 且 `time_range_label != "7d"`

**動作**

- `REFUSE(CLASSIFICATION, EXPLICIT_OUT_OF_SCOPE_TIME_RANGE)`

**說明**

- 「2330 有什麼重點？」→ 不拒絕，因為未明確指定時間，digest 預設 7 天
- 「2330 這一個月表現如何？」→ 拒絕，因為明確指定非 7 天

### R4b `OUT_OF_SCOPE_QUERY_KIND`

**條件**

- 命中 `OUT_OF_SCOPE_KEYWORDS`
- 或 query 明顯落在 digest 封閉集合外

**動作**

- `REFUSE(CLASSIFICATION, OUT_OF_SCOPE_QUERY_KIND)`

---

## 10. Retrieval Refusal 規則

### R5 `NO_EVIDENCE`

**條件**

- `len(retrieved_documents) == 0`

**動作**

- `REFUSE(EVIDENCE, NO_EVIDENCE)`

### R6 `STALE_EVIDENCE_ONLY`

**條件**

- 所有 retrieved documents 都不在近 7 天時間窗內

**動作**

- `REFUSE(EVIDENCE, STALE_EVIDENCE_ONLY)`

---

## 11. Governance Refusal 規則

### R7 `LOW_TRUST_SINGLE_SOURCE`

**條件**

- `len(evidence) == 1`
- 且 `high_trust_count == 0`

**動作**

- `REFUSE(GOVERNANCE, LOW_TRUST_SINGLE_SOURCE)`

### R8 `TOPIC_MISMATCH`

**條件**

- `topic_mismatch(query, evidence) == True`

**動作**

- `REFUSE(GOVERNANCE, TOPIC_MISMATCH)`

### R9 `INSUFFICIENT_TAG_COVERAGE`（v1.1 Dormant）

**條件**

- `classifier_tag_coverage == "insufficient"`

**動作**

- `REFUSE(GOVERNANCE, INSUFFICIENT_TAG_COVERAGE)`

**v1.1 狀態**：Dormant。由於 `MINIMUM_TAG_SET_BY_TOPIC` 皆為空集合（見 §5.9.1），此規則在 v1.1 永遠不觸發。enum、log 欄位保留，啟用條件見 §18.5、§18.6。

---

## 12. DEGRADED 規則

與 refusal 不同，DEGRADED 允許多個原因並存，因此 `degradedReasons` 必須是 list。

### D0 `SINGLE_HIGH_TRUST_SOURCE`

**條件**

- `len(evidence) == 1`
- `high_trust_count == 1`

**動作**

- `DEGRADED(["SINGLE_HIGH_TRUST_SOURCE"])`

**說明**

這條必須在 `cross_validated()` 之前處理，避免被誤標成 `WEAK_CROSS_VALIDATION`。

### D1 `WEAK_CROSS_VALIDATION`

**條件**

- `len(evidence) >= 2`
- 但 `cross_validated(evidence) == False`

### D2 `NO_HIGH_TRUST_SOURCE`

**條件**

- `high_trust_count == 0`
- （隱含 `len(evidence) >= 2`，因為 `(1, 0)` 已被 R7 攔截）

### D3 `BORDERLINE_FRESHNESS`

**條件**

- `freshness_strong(evidence, query_time) == False`
- 且 R6 未觸發（即 evidence 並非全部過期）

### D4 `EVIDENCE_CONFLICT`（v1.1 Dormant）

**條件**

- `evidence_conflict(governance_report) == True`

**v1.1 狀態**：Dormant。見 §5.8、§18.5。

### D5 `PARTIAL_TAG_COVERAGE`（v1.1 Dormant）

**條件**

- `classifier_tag_coverage == "partial"`

**v1.1 狀態**：Dormant。見 §5.9.1、§18.5。

---

## 13. NORMAL 規則

以下條件全滿足時，進入 NORMAL：

- ticker 可解析
- classifier 非 unknown
- query 未超 scope
- retrieved documents > 0
- evidence 不全過期
- 非低可信單一來源
- 非 topic mismatch
- 非 insufficient tag coverage（v1.1 因 R9 dormant，此項恆真）
- 無任何 degradedReasons

---

## 14. Enum Codes

### 14.1 `Outcome`

- `REFUSE`
- `DEGRADED`
- `NORMAL`

### 14.2 `RefusalCategory`

- `PARSE`
- `CLASSIFICATION`
- `EVIDENCE`
- `GOVERNANCE`

### 14.3 `RefusalReason`

- `UNRESOLVED_TICKER`
- `CLASSIFIER_UNKNOWN`
- `EXPLICIT_OUT_OF_SCOPE_TIME_RANGE`
- `OUT_OF_SCOPE_QUERY_KIND`
- `NO_EVIDENCE`
- `STALE_EVIDENCE_ONLY`
- `LOW_TRUST_SINGLE_SOURCE`
- `TOPIC_MISMATCH`
- `INSUFFICIENT_TAG_COVERAGE` *(v1.1 dormant)*

### 14.4 `DegradedCategory`

- `EVIDENCE`
- `FRESHNESS`
- `CONSISTENCY`
- `CLASSIFICATION`

### 14.5 `DegradedReason`

- `SINGLE_HIGH_TRUST_SOURCE`
- `WEAK_CROSS_VALIDATION`
- `NO_HIGH_TRUST_SOURCE`
- `BORDERLINE_FRESHNESS`
- `EVIDENCE_CONFLICT` *(v1.1 dormant)*
- `PARTIAL_TAG_COVERAGE` *(v1.1 dormant)*

---

## 15. Output Contract

### 15.1 REFUSE

```json
{
  "queryId": "uuid",
  "queryProfile": "single_stock_digest",
  "classifierSource": "llm",
  "outcome": "REFUSE",
  "confidenceLight": null,
  "confidenceScore": null,
  "refusalCategory": "EVIDENCE",
  "refusalReason": "NO_EVIDENCE",
  "warnings": [
    "No supporting evidence retrieved."
  ],
  "sources": []
}
```

REFUSE 的 `warnings` 應為 `refusalReason` 對應的 user-facing 訊息，一對一映射。

### 15.2 DEGRADED

```json
{
  "queryId": "uuid",
  "queryProfile": "single_stock_digest",
  "classifierSource": "llm",
  "outcome": "DEGRADED",
  "confidenceLight": "YELLOW",
  "confidenceScore": 0.56,
  "degradedReasons": [
    "WEAK_CROSS_VALIDATION",
    "NO_HIGH_TRUST_SOURCE"
  ],
  "warnings": [
    "Evidence is weakly cross-validated.",
    "No high-trust source found."
  ],
  "sources": ["..."]
}
```

### 15.3 NORMAL

```json
{
  "queryId": "uuid",
  "queryProfile": "single_stock_digest",
  "classifierSource": "llm",
  "outcome": "NORMAL",
  "confidenceLight": "GREEN",
  "confidenceScore": 0.84,
  "degradedReasons": [],
  "warnings": [],
  "sources": ["..."]
}
```

---

## 16. Query Log Requirements

即使被拒絕，也必須寫入 query log。

### 16.1 必存欄位

- `query_profile`
- `classifier_source`
- `classifier_raw_score`
- `classifier_tag_coverage`
- `time_range_explicit`
- `outcome`
- `refusal_category`
- `refusal_reason`
- `degraded_reasons`
- `structured_query_json`
- `response_json`
- `warnings`

### 16.2 建議欄位

- `user_feedback`
- `schema_version`（建議值：`"digest-v1.1"`）

---

## 17. Feedback Loop

Refusal 不能只是單向決策，系統必須預留使用者回報通道。

### 17.1 `user_feedback` 欄位

即使第一版 UI 尚未提供按鈕，schema 仍應預留：

- `REFUSED_CORRECT`
- `REFUSED_INCORRECT`
- `DEGRADED_CORRECT`
- `DEGRADED_INCORRECT`
- `null`

### 17.2 用途

- 收集 false refusal
- 收集 false degraded
- 作為閾值校準資料來源

---

## 18. 閾值校準

### 18.1 v1.1 所有閾值皆為暫定值

本文件中的 classifier score、evidence count、freshness 標準，皆屬 v1.1 暫定值。

### 18.2 首次校準條件

當 query log 累積：

- `>= 1000` 筆 digest queries
- 且有 `>= 100` 筆人工標註「應拒絕 / 不應拒絕」

應進行第一輪校準。

### 18.3 校準依據

- refusal precision / recall
- degraded precision / recall
- false refusal rate
- false normal rate

### 18.4 必須保留的原始訊號

- `classifier_raw_score`
- `classifier_tag_coverage`
- `outcome`
- `refusal_reason`
- `degraded_reasons`
- `user_feedback`

### 18.5 v1.1 Active vs Dormant Rules

**Active（v1.1 啟用）**

- R1, R2, R4, R4b, R5, R6, R7, R8
- D0, D1, D2, D3

**Dormant（schema 保留，v1.1 不觸發）**

- R9 `INSUFFICIENT_TAG_COVERAGE`
- D4 `EVIDENCE_CONFLICT`
- D5 `PARTIAL_TAG_COVERAGE`

Dormant rules 的 enum、log 欄位、UI 欄位**全部保留**。啟用時機與方式參見 §18.6。

### 18.6 Dormant 規則的升級觸發條件

**R9 / D5 (`classifier_tag_coverage`)** 啟用條件：

- 有 query log 資料顯示：當 classifier 無法覆蓋某 topic 的 key tags 時，digest 品質顯著下降
- `MINIMUM_TAG_SET_BY_TOPIC` 能以資料驅動方式填入具體 tag（不是直覺猜測）
- 啟用後 R9 / D5 的觸發率應先在 shadow mode 觀察 ≥ 200 筆 query

**D4 (`evidence_conflict`)** 啟用條件：

- governance layer 具備 `ExtractedFact` 結構化抽取能力
- conflict 偵測規則可由 `extracted_facts` 的確定性比較產生
- 啟用後 D4 的觸發率應先在 shadow mode 觀察 ≥ 200 筆 query

**啟用路徑**

- Dormant → Shadow（寫入 log 但不影響 outcome）
- Shadow → Active（影響 outcome，且 UI 顯示）

**回滾路徑**

- 若 Active 後 false rate 顯著偏高，可降回 Shadow 或 Dormant
- 回滾不需要修改 spec 契約

---

## 19. 參考偽代碼

```python
def early_refusal(query, classifier_result):
    if query.ticker is None:
        return REFUSE("PARSE", "UNRESOLVED_TICKER")

    if classifier_result.status == "unknown":
        return REFUSE("CLASSIFICATION", "CLASSIFIER_UNKNOWN")

    if query.time_range_explicit and query.time_range_label != "7d":
        return REFUSE("CLASSIFICATION", "EXPLICIT_OUT_OF_SCOPE_TIME_RANGE")

    if hits_out_of_scope_keywords(query.user_query):
        return REFUSE("CLASSIFICATION", "OUT_OF_SCOPE_QUERY_KIND")

    return None


def retrieval_refusal(query, retrieved_documents, query_time):
    if len(retrieved_documents) == 0:
        return REFUSE("EVIDENCE", "NO_EVIDENCE")

    if all(
        not within_7d(doc.published_at, query_time)
        for doc in retrieved_documents
    ):
        return REFUSE("EVIDENCE", "STALE_EVIDENCE_ONLY")

    return None


def governance_decision(query, governance_report, classifier_result, query_time):
    evidence = governance_report.evidence
    high_trust_count = sum(
        1 for item in evidence if item.source_tier == SourceTier.HIGH
    )

    # --- Refusal checks ---

    if len(evidence) == 1 and high_trust_count == 0:
        return REFUSE("GOVERNANCE", "LOW_TRUST_SINGLE_SOURCE")

    if topic_mismatch(query, evidence):
        return REFUSE("GOVERNANCE", "TOPIC_MISMATCH")

    # v1.1 dormant (see §5.9.1, §18.5); kept for v1.2+ activation
    if classifier_result.tag_coverage == "insufficient":
        return REFUSE("GOVERNANCE", "INSUFFICIENT_TAG_COVERAGE")

    # --- Degraded: single high-trust source ---

    if len(evidence) == 1 and high_trust_count == 1:
        return DEGRADED(["SINGLE_HIGH_TRUST_SOURCE"])

    # --- Degraded: multi-reason collection ---

    degraded_reasons = []

    if len(evidence) >= 2 and not cross_validated(evidence):
        degraded_reasons.append("WEAK_CROSS_VALIDATION")

    if high_trust_count == 0:
        degraded_reasons.append("NO_HIGH_TRUST_SOURCE")

    if not freshness_strong(evidence, query_time):
        degraded_reasons.append("BORDERLINE_FRESHNESS")

    # v1.1 dormant (see §5.8, §18.5); always False in v1.1
    if evidence_conflict(governance_report):
        degraded_reasons.append("EVIDENCE_CONFLICT")

    # v1.1 dormant (see §5.9.1, §18.5)
    if classifier_result.tag_coverage == "partial":
        degraded_reasons.append("PARTIAL_TAG_COVERAGE")

    if degraded_reasons:
        return DEGRADED(degraded_reasons)

    return NORMAL()


def evidence_conflict(governance_report) -> bool:
    # v1.1: dormant, see §5.8
    return False
```

---

## 20. 驗收標準

### 20.1 功能驗收

- digest 不再「幾乎永遠有答案」
- 無 ticker、無 evidence、分類不明、明確非 7 天範圍時可正確拒絕
- 單一高可信來源走 degraded，不會誤落為 `WEAK_CROSS_VALIDATION`

### 20.2 可觀測性驗收

- refusal / degraded / normal 都可寫入 query log
- `degradedReasons` 可同時多個
- `classifier_tag_coverage` 可回查
- Dormant 規則的對應欄位仍可在 log / response 中存在（值固定或 null）

### 20.3 契約驗收

- REFUSE 無 `confidenceLight` / `confidenceScore`
- NORMAL 一律 `GREEN`
- UI 不需用顏色猜 outcome

### 20.4 範圍驗收

- `len(ALLOWED_KEYWORD_TAGS) <= 10`（CI 測試)
- `sum(len(v) for v in MINIMUM_TAG_SET_BY_TOPIC.values()) <= 5`（CI 測試）
- digest path 不 import legacy query model
- digest path 不使用 `infer_intent_from_question_type()`
- digest path 不使用 `QUESTION_TYPE_FALLBACK_TOPIC_TAGS`

---

## 21. 最終定義

對 digest 產品線而言，真正的可信不是「加上一個 confidence light」，而是：

1. 系統知道什麼時候不該生成 digest
2. 系統知道什麼時候只能降級生成
3. 系統能把這些判斷完整寫入 query log 並回查

Refusal boundary 不是附屬功能，而是 digest 產品契約本身。

---

## 附錄 A：v1.1 → v1.2 升級檢核表

啟用 Dormant 規則時應確認：

- [ ] `MINIMUM_TAG_SET_BY_TOPIC` 以 query log 資料驅動填入（非猜測）
- [ ] 或 governance layer 具備 `ExtractedFact` 抽取能力
- [ ] 規則先在 Shadow mode 觀察 ≥ 200 筆 query
- [ ] false refusal / false degraded rate 可接受後再切 Active
- [ ] spec 版本號 bump（`schema_version: "digest-v1.2"`）
- [ ] 回滾路徑 (Active → Shadow → Dormant) 已驗證

---

*End of Spec v1.1 Final*
