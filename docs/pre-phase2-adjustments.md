# Phase 2 前置調整清單

> 目的：在改動 `query_data_hydrator.py` 之前，先確立輸入合約的語義，  
> 讓 Phase 2 的改動有清楚的基礎，且之後不需回頭修改 `core/` 層。

---

## 決策摘要

| 問題 | 決定 |
|------|------|
| intent 的地位 | Phase 2 起成為 hydrator 的**唯一**路由依據；question_type 保留給 Generation / Validation 層（Phase 4–5 前） |
| data_facets 的語義 | 拆為 `required_facets`（缺了→紅燈）與 `preferred_facets`（缺了→黃燈），各自在 hydration 和 validation 有不同行為 |
| topic_tags 的結構 | 拆為受控詞彙 `controlled_tags: list[TopicTag]`（可觀測、可路由）與原始關鍵字 `free_keywords: list[str]`（用於新聞搜尋） |
| 分析焦點層 | **暫緩**，Phase 5 設計 Generation template 時再議 |
| 觀測指標 | query_logs 加三個欄位；`_extract_topic_tags` 回傳 tag_source 信號 |

---

## 調整一：`DataFacet` 雙層化

### 背景

`INTENT_FACETS` 目前是單一 `frozenset`，無法區分「此 facet 缺失→直接降紅燈」與「此 facet 缺失→答案可能不完整但不阻斷」。  
Hydrator 和 Validation 兩層對「必要」vs「建議」的處理行為截然不同，必須在資料模型層就分清楚。

### 修改位置

**`src/llm_stock_system/core/models.py`**

```python
# 舊：單一 frozenset
INTENT_FACETS: dict[Intent, frozenset[DataFacet]] = { ... }

# 新：雙層宣告
@dataclass(frozen=True)
class FacetSpec:
    required: frozenset[DataFacet]    # 缺失 → confidence 壓到 0.25（紅燈）
    preferred: frozenset[DataFacet]   # 缺失 → confidence 降分但不壓死（黃燈範圍）

INTENT_FACET_SPECS: dict[Intent, FacetSpec] = {
    Intent.NEWS_DIGEST: FacetSpec(
        required=frozenset({DataFacet.NEWS}),
        preferred=frozenset({DataFacet.PRICE_HISTORY}),
    ),
    Intent.EARNINGS_REVIEW: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS}),
        preferred=frozenset({DataFacet.MONTHLY_REVENUE, DataFacet.NEWS}),
    ),
    Intent.VALUATION_CHECK: FacetSpec(
        required=frozenset({DataFacet.PE_VALUATION}),
        preferred=frozenset({DataFacet.PRICE_HISTORY, DataFacet.FINANCIAL_STATEMENTS, DataFacet.NEWS}),
    ),
    Intent.DIVIDEND_ANALYSIS: FacetSpec(
        required=frozenset({DataFacet.DIVIDEND}),
        preferred=frozenset({DataFacet.CASH_FLOW, DataFacet.BALANCE_SHEET, DataFacet.FINANCIAL_STATEMENTS}),
    ),
    Intent.FINANCIAL_HEALTH: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS}),
        preferred=frozenset({DataFacet.MONTHLY_REVENUE, DataFacet.NEWS}),
    ),
    Intent.TECHNICAL_VIEW: FacetSpec(
        required=frozenset({DataFacet.PRICE_HISTORY}),
        preferred=frozenset({DataFacet.MARGIN_DATA}),
    ),
    Intent.INVESTMENT_ASSESSMENT: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS, DataFacet.PE_VALUATION}),
        preferred=frozenset({DataFacet.DIVIDEND, DataFacet.NEWS, DataFacet.PRICE_HISTORY}),
    ),
}
```

**`src/llm_stock_system/core/models.py` — `StructuredQuery`**

```python
class StructuredQuery(BaseModel):
    ...
    intent: Intent = Intent.NEWS_DIGEST
    required_facets: set[DataFacet] = Field(default_factory=set)
    preferred_facets: set[DataFacet] = Field(default_factory=set)
    # data_facets 保留為 required ∪ preferred 的聯集，供現有程式碼過渡期使用
    data_facets: set[DataFacet] = Field(default_factory=set)
    ...
```

**`infer_data_facets()` 更新**

```python
def infer_data_facets(
    intent: Intent | str | None,
    extra_required: set[DataFacet] | None = None,
    extra_preferred: set[DataFacet] | None = None,
) -> tuple[set[DataFacet], set[DataFacet]]:
    """回傳 (required, preferred) 兩個集合"""
    spec = INTENT_FACET_SPECS.get(
        intent if isinstance(intent, Intent) else Intent(intent),
        FacetSpec(frozenset(), frozenset()),
    )
    required = set(spec.required) | (extra_required or set())
    preferred = set(spec.preferred) | (extra_preferred or set())
    return required, preferred
```

### Hydrator 行為差異（Phase 2 實作時參照）

| facet 類別 | sync 失敗時的處理 | 記錄方式 |
|-----------|----------------|---------|
| required  | 記錄 WARNING，寫入 `facet_miss_list`，不中斷流程 | query_logs.facet_miss_list |
| preferred | 靜默略過，不記錄 | — |

### Validation 行為差異（Phase 4 實作時參照）

| facet 類別 | evidence 缺失時 | confidence 影響 |
|-----------|--------------|----------------|
| required  | `confidence_score = min(score, 0.25)` | 紅燈 |
| preferred | `confidence_score -= 0.1`（可累加，最多 -0.3） | 仍可能黃燈 |

---

## 調整二：`TopicTag` 受控詞彙 + `topic_tags` 欄位拆分

### 背景

現在的 `topic_tags: list[str]` 同時混入了「分類標籤」（航運）和「搜尋關鍵字」（SCFI），導致：
- 無法統計 tag 分布（分類標籤和搜尋詞的 cardinality 不同）
- Retrieval Layer 不知道哪個是路由信號、哪個是搜尋詞

### 修改位置

**`src/llm_stock_system/core/enums.py`**

```python
class TopicTag(str, Enum):
    """受控詞彙。每個值代表一個可觀測、可路由的分析主題。"""
    SHIPPING       = "航運"
    ELECTRICITY    = "電價"
    MACRO          = "總經"
    GUIDANCE       = "法說"
    TECHNICAL      = "技術面"
    MARGIN_FLOW    = "籌碼"
    SEMICON_EQUIP  = "半導體設備"
    EV             = "電動車"
    AI             = "AI"
    DIVIDEND       = "股利"
    REVENUE        = "月營收"
    GROSS_MARGIN   = "毛利率"
    VALUATION      = "本益比"
    FUNDAMENTAL    = "基本面"
    CASH_FLOW      = "現金流"
    DEBT           = "負債"
    LISTING        = "上市"
```

**`src/llm_stock_system/core/models.py` — `StructuredQuery`**

```python
class StructuredQuery(BaseModel):
    ...
    controlled_tags: list[TopicTag] = Field(default_factory=list)
    # ^ 受控詞彙，用於觀測、Retrieval 路由、Validation 策略選擇
    free_keywords: list[str] = Field(default_factory=list)
    # ^ 從查詢文字中命中的原始關鍵字，用於新聞搜尋的 search_terms
    topic_tags: list[str] = Field(default_factory=list)
    # ^ 過渡期保留：= [t.value for t in controlled_tags] + free_keywords
    ...
```

**`_TOPIC_TAG_KEYWORD_MAP` 型別對應更新**

```python
# InputLayer 中，key 從 str 改為 TopicTag
_TOPIC_TAG_KEYWORD_MAP: dict[TopicTag, tuple[str, ...]] = {
    TopicTag.SHIPPING:      ("紅海", "航線", "SCFI", "scfi", "運價", "集運", "航運", "散裝"),
    TopicTag.ELECTRICITY:   ("工業電價", "電價", "調漲", "漲價", "電費", "用電大戶", "節能", "節電"),
    TopicTag.MACRO:         ("CPI", "cpi", "通膨", "利率", "殖利率", "美債", "降息", "升息", "聯準會"),
    TopicTag.GUIDANCE:      ("法說", "法說會", "營運指引", "指引", "財測", "展望"),
    TopicTag.TECHNICAL:     ("RSI", "rsi", "KD", "kd", "MACD", "macd", "布林通道", "均線", "乖離", "超買", "超賣"),
    TopicTag.MARGIN_FLOW:   ("融資", "融券", "信用交易", "籌碼", "季線", "60日線", "60MA", "ma60"),
    TopicTag.SEMICON_EQUIP: ("半導體設備", "設備族群", "設備股", "ASML", "asml", "艾司摩爾"),
    TopicTag.EV:            ("電動車", "EV", "ev", "電池", "供應鏈", "普及率"),
    TopicTag.AI:            ("AI", "ai", "伺服器", "server", "AI伺服器"),
    TopicTag.DIVIDEND:      ("股利", "配息", "現金股利", "除息", "殖利率", "填息", "填權"),
    TopicTag.REVENUE:       ("月營收", "累計營收", "年增", "月增", "MoM", "mom", "創新高"),
    TopicTag.GROSS_MARGIN:  ("毛利率", "毛利", "營業毛利", "經營效率"),
    TopicTag.VALUATION:     ("本益比", "P/E", "p/e", "PE ratio", "估值"),
    TopicTag.FUNDAMENTAL:   ("基本面", "體質", "營運"),
    TopicTag.CASH_FLOW:     ("自由現金流", "FCF", "fcf", "營業現金流", "現金流"),
    TopicTag.DEBT:          ("負債比率", "負債比", "負債總額", "槓桿"),
    TopicTag.LISTING:       ("轉上市", "上市", "掛牌", "IPO", "ipo", "新上市"),
}
```

**`_extract_topic_tags` 更新為回傳雙欄位**

```python
def _extract_topic_tags(
    self, query: str, question_type: str
) -> tuple[list[TopicTag], list[str]]:
    """
    回傳 (controlled_tags, free_keywords)。
    controlled_tags: 命中的受控 TopicTag
    free_keywords:   命中的原始關鍵字（供 Retrieval Layer 搜尋用）
    """
    lowered = query.lower()
    compacted = self._compact_query(query)
    seen_keywords: set[str] = set()
    matched_tags: list[TopicTag] = []
    matched_keywords: list[str] = []

    for tag, keywords in self._TOPIC_TAG_KEYWORD_MAP.items():
        kws_hit = [
            kw for kw in keywords
            if (kw.lower() in lowered or kw in query
                or kw.lower().replace(" ", "") in compacted)
            and kw not in seen_keywords
        ]
        if kws_hit:
            matched_tags.append(tag)
            seen_keywords.update(kws_hit)
            matched_keywords.extend(kws_hit)

    if matched_tags:
        return matched_tags, matched_keywords

    # Fallback：從 QUESTION_TYPE_FALLBACK_TOPIC_TAGS 取回舊標籤，
    # 無法對應到 TopicTag 的保留在 free_keywords
    fallback = self.QUESTION_TYPE_FALLBACK_TOPIC_TAGS.get(question_type, ())
    return [], list(fallback)
```

---

## 調整三：`query_logs` 加觀測欄位

### 背景

沒有觀測指標就無法驗證遷移是否正確，以及評估 tag 覆蓋率與 facet 缺失率。

### 修改位置

**`db/sql/`（新增 migration SQL）**

```sql
-- migration: add intent observability columns to query_logs
ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS intent           VARCHAR(32),
    ADD COLUMN IF NOT EXISTS controlled_tags  TEXT[],
    ADD COLUMN IF NOT EXISTS facet_miss_list  TEXT[],
    ADD COLUMN IF NOT EXISTS tag_source       VARCHAR(16);
    -- tag_source: 'matched' | 'fallback' | 'empty'
```

### 三個核心查詢（建立後可直接使用）

**① question_type → intent 映射分布**（確認流量集中在哪些 intent）
```sql
SELECT question_type, intent, COUNT(*) AS cnt
FROM query_logs
WHERE intent IS NOT NULL
GROUP BY question_type, intent
ORDER BY cnt DESC;
```

**② Facet 缺失率**（哪些 facet sync 最常失敗）
```sql
SELECT unnest(facet_miss_list) AS missing_facet, COUNT(*) AS cnt
FROM query_logs
WHERE facet_miss_list IS NOT NULL AND array_length(facet_miss_list, 1) > 0
GROUP BY missing_facet
ORDER BY cnt DESC;
```

**③ Tag 命中率**（`_TOPIC_TAG_KEYWORD_MAP` 的覆蓋程度）
```sql
SELECT intent, tag_source, COUNT(*) AS cnt
FROM query_logs
WHERE intent IS NOT NULL
GROUP BY intent, tag_source
ORDER BY intent, tag_source;
```
> `tag_source = 'fallback'` → keyword map 未覆蓋，需擴充  
> `tag_source = 'empty'` → 連 fallback 都沒有，需補 `QUESTION_TYPE_FALLBACK_TOPIC_TAGS`

---

## 調整四：確立 intent 的權威性邊界

### 規則

```
               ┌─────────────────────────────────┐
               │           StructuredQuery         │
               │                                   │
  輸入合約     │  intent          ← Phase 2 起授  │
  (Phase 2+)  │  required_facets ← Hydrator 讀這  │
               │  preferred_facets                 │
               │  controlled_tags ← Retrieval 讀這 │
               │  free_keywords   ← 新聞搜尋用這   │
               │                                   │
  過渡期保留   │  question_type   ← Gen / Val 仍讀 │
  (Phase 4–5  │  data_facets     ← 聯集，向下相容 │
   前移除)    │  topic_tags      ← 聯集，向下相容 │
               └─────────────────────────────────┘
```

**Phase 2 的 `query_data_hydrator.py` 只能讀：**
- `query.intent`
- `query.required_facets`
- `query.preferred_facets`
- `query.ticker`, `query.time_range_days`（業務必要）

**Phase 2 的 hydrator 不得讀：**
- `query.question_type`（讀了就是沒有解耦）

---

## 調整執行順序

```
1. enums.py        → 新增 TopicTag enum
2. models.py       → 新增 FacetSpec dataclass + INTENT_FACET_SPECS
                   → StructuredQuery 加 required_facets / preferred_facets
                                              / controlled_tags / free_keywords
                   → 更新 model_validator 呼叫 infer_data_facets() 的方式
3. input_layer.py  → _TOPIC_TAG_KEYWORD_MAP key 改為 TopicTag
                   → _extract_topic_tags 改為回傳 tuple[list[TopicTag], list[str]]
                   → parse() 填入 controlled_tags / free_keywords / tag_source
4. db migration    → query_logs 加三欄位
5. tests           → 更新 test_intent_metadata.py 驗證新欄位
                   → 新增 test_topic_tag_coverage.py 驗證 tag hit rate
```

---

## 完成標準（Definition of Done）

進入 Phase 2 之前，以下條件必須全部成立：

- [ ] `StructuredQuery` 同時有 `required_facets` 和 `preferred_facets`
- [ ] `TopicTag` enum 已定義，`_TOPIC_TAG_KEYWORD_MAP` key 已改為 `TopicTag`
- [ ] `_extract_topic_tags` 回傳 `tuple[list[TopicTag], list[str]]`
- [ ] `StructuredQuery.controlled_tags` 和 `free_keywords` 有值（不是空 list）
- [ ] `query_logs` migration SQL 已準備好（不需 apply，有 SQL 即可）
- [ ] `test_intent_metadata.py` 驗證 `controlled_tags` 和 `required_facets` 的正確性
- [ ] 所有現有測試仍然通過（83 passed）
