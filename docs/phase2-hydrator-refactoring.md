# Phase 2: QueryDataHydrator Refactoring
## From question_type-based to Intent + DataFacet-based Routing

**Status**: Completed  
**Target Release**: Post Phase 0+1 validation  
**Estimated Effort**: 3-4 days (implementation + testing)  
**Actual Effort**: 1 implementation pass + targeted regression coverage  
**Date Completed**: April 14, 2026  
**Risk Level**: Medium (affects data retrieval path, but no breaking API changes)

---

## 1. Executive Summary

The `QueryDataHydrator` currently uses 28 question_type values to determine which market data (price history, financial statements, dividends, etc.) to sync before the retrieval and generation layers run.

Phase 2 refactors this class to use the new Intent-based routing system (7 intents) + DataFacet requirements, reducing decision branches from 28 to 7 while making facet requirements **explicit and observable**.

### Key Outcomes
- **Reduced Complexity**: 28 question_type branches → 7 intent-based routing rules
- **Explicit Requirements**: Required vs preferred facets tracked in `StructuredQuery`
- **Observability**: New `facet_miss_list` in query_logs tracks which facets failed to sync
- **Maintainability**: Adding new question types no longer requires modifying hydrator; just map to existing Intent + facets

---

## 2. Current State Analysis

### 2.1 Existing Architecture

**File**: `src/llm_stock_system/services/query_data_hydrator.py` (231 lines)

**Current Routing Logic**:
- Three hardcoded question_type sets: `PRICE_RELATED_QUESTION_TYPES`, `FUNDAMENTAL_QUESTION_TYPES`, `NEWS_RELATED_QUESTION_TYPES`
- 13 separate `if query.question_type in {...}` or `if query.question_type == "..."` branches
- Each branch calls 1-3 gateway sync methods (e.g., sync_price_history, sync_financial_statements)

**Example Current Branch** (line 93-97):
```python
if query.question_type == "monthly_revenue_yoy_review":
    self._safe_call(self._gateway.sync_monthly_revenue_points, ticker)

if query.question_type == "listing_revenue_review":
    self._safe_call(self._gateway.sync_monthly_revenue_points, ticker)
```

**Problem**: Both branch on the same data (monthly_revenue) but are separate conditions. Under intent routing, both map to `Intent.EARNINGS_REVIEW` → `preferred_facets: {DataFacet.MONTHLY_REVENUE}`, unified into one sync rule.

### 2.2 Gateway Methods (Available Sync Operations)

From `_gateway` interface (implied by current code):

| Gateway Method | DataFacet | Purpose | Time Range |
|---|---|---|---|
| `sync_stock_info()` | — | Stock metadata (name, industry, type) | current snapshot |
| `sync_price_history(ticker, start, end)` | `PRICE_HISTORY` | OHLCV data | variable (default: 120d) |
| `sync_financial_statements(ticker, start, end)` | `FINANCIAL_STATEMENTS` | P&L, income data | variable (default: multi-year) |
| `sync_monthly_revenue_points(ticker)` | `MONTHLY_REVENUE` | MoM/YoY revenue | last ~24 months |
| `sync_dividend_policies(ticker, start, end)` | `DIVIDEND` | Cash/stock distributions | variable (default: multi-year) |
| `sync_balance_sheet_items(ticker, start, end)` | `BALANCE_SHEET` | Assets, liabilities, equity | variable (default: multi-year) |
| `sync_cash_flow_statements(ticker, start, end)` | `CASH_FLOW` | Operating/investing/financing CF | variable (default: multi-year) |
| `sync_pe_valuation_points(ticker)` | `PE_VALUATION` | P/E ratio history | last ~24 months |
| `sync_margin_purchase_short_sale(ticker, start, end)` | `MARGIN_DATA` | Margin/short sale flows | variable (default: 180d) |
| `sync_stock_news(ticker, start, end)` | `NEWS` | News articles | variable (default: 30d) |
| `sync_query_news(query)` | `NEWS` | Smart search news (optional) | query-dependent |

### 2.3 Existing Hidden Dependencies

**Time Range Calculations** (lines 78-82):
```python
history_years = max((query.time_range_days + 364) // 365, 2)
price_start = today - timedelta(days=max(query.time_range_days, 120))
fundamentals_start = date(today.year - history_years, 1, 1)
long_term_start = date(today.year - 3, 1, 1)
dividend_start = max(date(today.year - history_years, 1, 1), today - timedelta(days=730))
```

These must be preserved and made **facet-aware** in Phase 2.

---

## 3. Intent → DataFacet Mapping (From models.py)

### INTENT_FACET_SPECS Mapping

| Intent | Required Facets | Preferred Facets |
|---|---|---|
| `NEWS_DIGEST` | `NEWS` | `PRICE_HISTORY` |
| `EARNINGS_REVIEW` | `FINANCIAL_STATEMENTS` | `MONTHLY_REVENUE`, `NEWS` |
| `VALUATION_CHECK` | `PE_VALUATION` | `PRICE_HISTORY`, `FINANCIAL_STATEMENTS`, `NEWS` |
| `DIVIDEND_ANALYSIS` | `DIVIDEND` | `CASH_FLOW`, `BALANCE_SHEET`, `FINANCIAL_STATEMENTS` |
| `FINANCIAL_HEALTH` | `FINANCIAL_STATEMENTS` | `MONTHLY_REVENUE`, `NEWS` |
| `TECHNICAL_VIEW` | `PRICE_HISTORY` | `MARGIN_DATA` |
| `INVESTMENT_ASSESSMENT` | `FINANCIAL_STATEMENTS`, `PE_VALUATION` | `DIVIDEND`, `NEWS`, `PRICE_HISTORY` |

### Derivation from question_type

From `QUESTION_TYPE_TO_INTENT` (models.py lines 22-50):

```python
# NEWS_DIGEST (8 question_types)
market_summary, theme_impact_review, shipping_rate_impact_review,
electricity_cost_impact_review, macro_yield_sentiment_review,
guidance_reaction_review, listing_revenue_review → NEWS_DIGEST → NEWS + PRICE_HISTORY

# EARNINGS_REVIEW (4 types)
earnings_summary, eps_dividend_review, monthly_revenue_yoy_review, 
margin_turnaround_review → EARNINGS_REVIEW → FINANCIAL_STATEMENTS + MONTHLY_REVENUE

# VALUATION_CHECK (4 types)
pe_valuation_review, fundamental_pe_review, price_range, price_outlook
→ VALUATION_CHECK → PE_VALUATION + PRICE_HISTORY + FINANCIAL_STATEMENTS

# DIVIDEND_ANALYSIS (4 types)
dividend_yield_review, ex_dividend_performance, 
fcf_dividend_sustainability_review, debt_dividend_safety_review
→ DIVIDEND_ANALYSIS → DIVIDEND + CASH_FLOW + BALANCE_SHEET

# FINANCIAL_HEALTH (3 types)
profitability_stability_review, gross_margin_comparison_review, revenue_growth_review
→ FINANCIAL_HEALTH → FINANCIAL_STATEMENTS + MONTHLY_REVENUE

# TECHNICAL_VIEW (2 types)
technical_indicator_review, season_line_margin_review
→ TECHNICAL_VIEW → PRICE_HISTORY + MARGIN_DATA

# INVESTMENT_ASSESSMENT (3 types)
investment_support, risk_review, announcement_summary
→ INVESTMENT_ASSESSMENT → FINANCIAL_STATEMENTS + PE_VALUATION + DIVIDEND
```

---

## 4. Sync Strategy: DataFacet → Gateway Method

### 4.1 Facet Sync Mapping

Create a new internal constant in `QueryDataHydrator`:

```python
FACET_SYNC_MAP: dict[DataFacet, FacetSyncRule] = {
    DataFacet.PRICE_HISTORY: FacetSyncRule(
        gateway_method="sync_price_history",
        requires_dates=True,
        default_window_days=120,
        min_window_days=60,
        priority="high"
    ),
    DataFacet.FINANCIAL_STATEMENTS: FacetSyncRule(
        gateway_method="sync_financial_statements",
        requires_dates=True,
        default_window_years=2,
        min_window_years=1,
        priority="high"
    ),
    DataFacet.MONTHLY_REVENUE: FacetSyncRule(
        gateway_method="sync_monthly_revenue_points",
        requires_dates=False,  # method auto-fetches last ~24 months
        priority="medium"
    ),
    DataFacet.DIVIDEND: FacetSyncRule(
        gateway_method="sync_dividend_policies",
        requires_dates=True,
        default_window_years=2,
        priority="high"
    ),
    DataFacet.BALANCE_SHEET: FacetSyncRule(
        gateway_method="sync_balance_sheet_items",
        requires_dates=True,
        default_window_years=3,
        priority="medium"
    ),
    DataFacet.CASH_FLOW: FacetSyncRule(
        gateway_method="sync_cash_flow_statements",
        requires_dates=True,
        default_window_years=3,
        priority="medium"
    ),
    DataFacet.PE_VALUATION: FacetSyncRule(
        gateway_method="sync_pe_valuation_points",
        requires_dates=False,  # auto-fetches history
        priority="high"
    ),
    DataFacet.MARGIN_DATA: FacetSyncRule(
        gateway_method="sync_margin_purchase_short_sale",
        requires_dates=True,
        default_window_days=180,
        priority="low"
    ),
    DataFacet.NEWS: FacetSyncRule(
        gateway_method="sync_stock_news",
        requires_dates=True,
        default_window_days=30,
        fallback_method="sync_query_news",
        priority="high"
    ),
}
```

### 4.2 Time Range Calculation per Facet

**Smart Windowing Logic**:
```python
def _compute_facet_window(
    facet: DataFacet,
    query: StructuredQuery,
    today: date
) -> tuple[date, date] | None:
    """
    Compute optimal start/end dates for a given facet.
    
    Logic:
    - Use query.time_range_days as base window
    - But enforce minimum and maximum reasonable windows per facet
    - Some facets don't need dates (MONTHLY_REVENUE, PE_VALUATION)
    
    Returns:
    - (start_date, end_date) for facets requiring dates
    - None for auto-fetch facets
    """
    if facet == DataFacet.PRICE_HISTORY:
        window_days = max(query.time_range_days, 120)
        return today - timedelta(days=window_days), today
    
    elif facet in (DataFacet.FINANCIAL_STATEMENTS, DataFacet.DIVIDEND, 
                   DataFacet.BALANCE_SHEET, DataFacet.CASH_FLOW):
        history_years = max((query.time_range_days + 364) // 365, 2)
        return date(today.year - history_years, 1, 1), today
    
    elif facet in (DataFacet.BALANCE_SHEET, DataFacet.CASH_FLOW):
        # Long-term view for debt/CF analysis
        return date(today.year - 3, 1, 1), today
    
    elif facet == DataFacet.MARGIN_DATA:
        return today - timedelta(days=180), today
    
    elif facet == DataFacet.NEWS:
        window_days = max(query.time_range_days, 30)
        return today - timedelta(days=window_days), today
    
    elif facet in (DataFacet.MONTHLY_REVENUE, DataFacet.PE_VALUATION):
        return None  # Auto-fetch methods; no window needed
    
    return None
```

---

## 5. Failure Handling & Observability

### 5.1 Facet Success/Failure Tracking

**New Return Type**:
```python
@dataclass
class FacetSyncResult:
    facet: DataFacet
    success: bool
    exception: Exception | None = None
    duration_ms: float = 0
    rows_synced: int | None = None
```

**Hydrate Return Type** (enhanced):
```python
@dataclass
class HydrationResult:
    query_id: str
    synced_facets: set[DataFacet]
    failed_facets: dict[DataFacet, str]  # facet → error message
    facet_miss_list: list[str]  # list of required facets that failed
    total_duration_ms: float
```

### 5.2 Missing Facet Decision Logic

**Required Facets**:
- If sync fails → Add to `facet_miss_list`
- Warning → "Required facet X not available"
- Proceeding decision: **YES** (generation layer will handle gracefully)

**Preferred Facets**:
- If sync fails → Silently skip
- No warning
- Proceeding decision: **YES** (expected behavior)

**Confidence Adjustment**:
- Required facet missing → `confidence_light` becomes RED or YELLOW
- Preferred facets missing → confidence light unchanged

### 5.3 Storing Results in query_logs

From `db/sql/003_query_log_observability.sql`:

```sql
-- Existing: facet_miss_list TEXT[] already defined
-- Update hydrate() to populate this at query completion

INSERT INTO query_logs (
    query_id, 
    intent, 
    controlled_tags, 
    facet_miss_list,  -- Array of required facets that failed
    tag_source
) VALUES (...)
```

---

## 6. Refactored Architecture

### 6.1 New hydrate() Signature

**Before**:
```python
def hydrate(self, query: StructuredQuery) -> None:
    if not query.ticker:
        return
    # ... logic ...
```

**After**:
```python
def hydrate(self, query: StructuredQuery) -> HydrationResult:
    """
    Sync market data based on query intent + required/preferred facets.
    
    Args:
        query: StructuredQuery with intent, required_facets, preferred_facets
    
    Returns:
        HydrationResult with synced_facets, failed_facets, facet_miss_list
    """
    if not query.ticker:
        return HydrationResult(
            query_id=query.query_id or str(uuid4()),
            synced_facets=set(),
            failed_facets={},
            facet_miss_list=[],
            total_duration_ms=0.0
        )
    
    result = HydrationResult(
        query_id=query.query_id or str(uuid4()),
        synced_facets=set(),
        failed_facets={},
        facet_miss_list=[],
        total_duration_ms=0.0
    )
    
    today = datetime.now(timezone.utc).date()
    start_time = time.time()
    
    # Sync stock metadata (always)
    self._safe_call(self._gateway.sync_stock_info)
    
    # Get all facets to sync (required + preferred)
    facets_to_sync = query.required_facets | query.preferred_facets
    
    tickers = [query.ticker]
    if query.comparison_ticker and query.comparison_ticker not in tickers:
        tickers.append(query.comparison_ticker)
    
    # Sync each facet for each ticker
    for ticker in tickers:
        for facet in facets_to_sync:
            is_required = facet in query.required_facets
            facet_result = self._sync_facet(facet, ticker, query, today)
            
            if facet_result.success:
                result.synced_facets.add(facet)
            else:
                result.failed_facets[facet] = str(facet_result.exception)
                if is_required:
                    result.facet_miss_list.append(facet.value)
    
    result.total_duration_ms = (time.time() - start_time) * 1000
    return result
```

### 6.2 New _sync_facet() Method

```python
def _sync_facet(
    self, 
    facet: DataFacet, 
    ticker: str, 
    query: StructuredQuery, 
    today: date
) -> FacetSyncResult:
    """
    Sync a single DataFacet for a single ticker.
    
    Handles:
    - Date range computation
    - Gateway method dispatch
    - Exception capturing
    - Duration tracking
    """
    facet_rule = self.FACET_SYNC_MAP.get(facet)
    if not facet_rule:
        return FacetSyncResult(
            facet=facet, 
            success=False, 
            exception=ValueError(f"Unknown facet: {facet}")
        )
    
    try:
        start_time = time.time()
        
        # Compute date window if needed
        date_range = self._compute_facet_window(facet, query, today)
        
        # Dispatch to appropriate gateway method
        if facet_rule.requires_dates and date_range:
            start_date, end_date = date_range
            getattr(self._gateway, facet_rule.gateway_method)(
                ticker, start_date, end_date
            )
        else:
            # No date args needed (e.g., sync_monthly_revenue_points)
            if facet == DataFacet.NEWS and query.intent in [Intent.NEWS_DIGEST]:
                # Try smart search first
                sync_method = getattr(self._gateway, 'sync_query_news', None)
                if callable(sync_method):
                    sync_method(query)
                else:
                    # Fall back to date-range search
                    date_range = self._compute_facet_window(facet, query, today)
                    if date_range:
                        start_date, end_date = date_range
                        self._gateway.sync_stock_news(ticker, start_date, end_date)
            else:
                getattr(self._gateway, facet_rule.gateway_method)(ticker)
        
        duration_ms = (time.time() - start_time) * 1000
        return FacetSyncResult(
            facet=facet,
            success=True,
            duration_ms=duration_ms
        )
    
    except Exception as e:
        return FacetSyncResult(
            facet=facet,
            success=False,
            exception=e,
            duration_ms=(time.time() - start_time) * 1000
        )
```

### 6.3 Deprecated Methods

**Remove** (no longer needed):
- `_should_sync_query_news()` → Moved to facet logic
- `_sync_query_news()` → Moved to facet logic
- `_hydrate_ticker()` → Refactored into `_sync_facet()`
- Hardcoded question_type sets: `PRICE_RELATED_QUESTION_TYPES`, etc.

**Keep** (still useful):
- `schedule_follow_up()` and `_run_follow_up_bundle()` → Works with new HydrationResult
- `_safe_call()` → Error handling unchanged
- `_mark_follow_up_started()`, `_iter_tickers()` → Utility methods unchanged

---

## 7. Data Structure Changes

### 7.1 New Imports

```python
from dataclasses import dataclass
import time
from uuid import uuid4

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import StructuredQuery, HydrationResult
```

### 7.2 New Dataclasses

```python
@dataclass(frozen=True)
class FacetSyncRule:
    """Configuration for syncing a single DataFacet."""
    gateway_method: str  # e.g., "sync_price_history"
    requires_dates: bool = False
    default_window_days: int | None = None
    default_window_years: int | None = None
    min_window_days: int | None = None
    min_window_years: int | None = None
    fallback_method: str | None = None
    priority: str = "medium"  # high, medium, low


@dataclass
class FacetSyncResult:
    """Result of attempting to sync one facet."""
    facet: DataFacet
    success: bool
    exception: Exception | None = None
    duration_ms: float = 0.0
    rows_synced: int | None = None
```

### 7.3 Updates to StructuredQuery

**No changes needed** — already has:
- `required_facets: set[DataFacet]`
- `preferred_facets: set[DataFacet]`
- `intent: Intent`

### 7.4 Updates to query_logs (DB)

**Already done** in Phase 0+1:
```sql
ALTER TABLE query_logs ADD COLUMN facet_miss_list TEXT[];
```

**Populate in pipeline** (Presentation Layer responsible):
```python
# When logging query completion:
query_log.facet_miss_list = hydration_result.facet_miss_list
```

---

## 8. Test Strategy

### 8.1 Test Structure

**File**: `tests/test_query_data_hydrator_phase2.py` (new)

**Test Categories**:
1. **Facet-to-Intent Mapping** (4 tests)
   - Each Intent syncs correct required + preferred facets
   
2. **Date Range Computation** (6 tests)
   - PRICE_HISTORY: Respects time_range_days
   - FINANCIAL_STATEMENTS: Multi-year window
   - NEWS: Dynamic window based on query
   - BALANCE_SHEET/CASH_FLOW: 3-year window
   - MARGIN_DATA: 180-day window
   - Auto-fetch facets (MONTHLY_REVENUE, PE_VALUATION): None
   
3. **Facet-to-Gateway Dispatch** (9 tests)
   - Each facet calls correct gateway method
   - Date args passed correctly
   - Single vs multi-ticker handling
   
4. **Failure Handling** (5 tests)
   - Required facet failure → added to facet_miss_list
   - Preferred facet failure → ignored
   - HydrationResult populated correctly
   - Exception captured (not raised)
   
5. **Intent-Specific Scenarios** (7 tests)
   - NEWS_DIGEST: NEWS required, PRICE_HISTORY preferred
   - VALUATION_CHECK: PE_VALUATION required, FINANCIAL_STATEMENTS + PRICE_HISTORY preferred
   - DIVIDEND_ANALYSIS: DIVIDEND required, CASH_FLOW + BALANCE_SHEET preferred
   - (etc. for all 7 intents)
   
6. **Backward Compatibility** (2 tests)
   - Old question_type mappings still work
   - StructuredQuery auto-populates intent/facets from question_type

### 8.2 Test Examples

**Example 1: VALUATION_CHECK maps to correct facets**
```python
def test_valuation_check_intent_syncs_all_facets(self):
    query = StructuredQuery(
        user_query="台積電 PE比?",
        ticker="2330",
        intent=Intent.VALUATION_CHECK,
        required_facets={DataFacet.PE_VALUATION},
        preferred_facets={
            DataFacet.PRICE_HISTORY,
            DataFacet.FINANCIAL_STATEMENTS,
            DataFacet.NEWS,
        }
    )
    
    gateway_mock = Mock()
    hydrator = QueryDataHydrator(gateway_mock)
    result = hydrator.hydrate(query)
    
    # Verify all 4 facets attempted
    self.assertEqual(len(result.synced_facets) + len(result.failed_facets), 4)
    
    # Verify sync_pe_valuation_points called
    gateway_mock.sync_pe_valuation_points.assert_called_once_with("2330")
    
    # Verify sync_price_history called with correct date range
    gateway_mock.sync_price_history.assert_called_once()
    args = gateway_mock.sync_price_history.call_args
    self.assertEqual(args[0][0], "2330")  # ticker
    self.assertIsInstance(args[0][1], date)  # start_date
    self.assertIsInstance(args[0][2], date)  # end_date
```

**Example 2: Required facet failure recorded in facet_miss_list**
```python
def test_required_facet_failure_recorded(self):
    query = StructuredQuery(
        user_query="長榮紅海運費",
        ticker="2603",
        intent=Intent.NEWS_DIGEST,
        required_facets={DataFacet.NEWS},
        preferred_facets={DataFacet.PRICE_HISTORY},
    )
    
    gateway_mock = Mock()
    gateway_mock.sync_stock_news.side_effect = ConnectionError("API down")
    hydrator = QueryDataHydrator(gateway_mock)
    
    result = hydrator.hydrate(query)
    
    # NEWS is required, so failure is recorded
    self.assertIn("news", result.facet_miss_list)
    self.assertIn(DataFacet.NEWS, result.failed_facets)
    
    # PRICE_HISTORY is preferred, so no entry in facet_miss_list
    # even if it also failed
    self.assertEqual(result.facet_miss_list, ["news"])
```

**Example 3: Backward compatibility with question_type**
```python
def test_backward_compatibility_question_type_to_intent_mapping(self):
    # Old style: just question_type, no explicit intent
    query = StructuredQuery(
        user_query="月營收",
        ticker="2330",
        question_type="monthly_revenue_yoy_review"
        # intent not specified, auto-populated by model_validator
    )
    
    self.assertEqual(query.intent, Intent.EARNINGS_REVIEW)
    self.assertEqual(query.required_facets, {DataFacet.FINANCIAL_STATEMENTS})
    self.assertIn(DataFacet.MONTHLY_REVENUE, query.preferred_facets)
```

### 8.3 Test Coverage Targets

- **Line coverage**: ≥90% (all facet paths exercised)
- **Branch coverage**: ≥85% (all success/failure paths)
- **Intent coverage**: 100% (all 7 intents tested)
- **Gateway method coverage**: 100% (all 10 sync methods tested)

---

## 9. File-by-File Modification Checklist

### Phase 2 Deliverables

- [ ] **src/llm_stock_system/services/query_data_hydrator.py**
  - [ ] Add imports: `dataclass`, `time`, `uuid4`, `DataFacet`, `Intent`
  - [ ] Add `FacetSyncRule` and `FacetSyncResult` dataclasses
  - [ ] Add `FACET_SYNC_MAP` constant (9 entries)
  - [ ] Implement `_compute_facet_window()` method
  - [ ] Refactor `hydrate()` → returns `HydrationResult`
  - [ ] Implement `_sync_facet()` method (core routing logic)
  - [ ] Delete `_hydrate_ticker()` method
  - [ ] Delete `_should_sync_query_news()`, `_sync_query_news()` methods
  - [ ] Delete hardcoded question_type sets
  - [ ] Update `schedule_follow_up()` to work with `HydrationResult`
  - [ ] Update docstrings

- [ ] **src/llm_stock_system/core/models.py**
  - [ ] Add `HydrationResult` dataclass (if not already present)
  - [ ] Verify `StructuredQuery` has required_facets, preferred_facets, intent
  - [ ] Verify model_validator auto-populates intent/facets from question_type

- [ ] **tests/test_query_data_hydrator_phase2.py** (new file)
  - [ ] Import test fixtures and mocks
  - [ ] 24 tests as per section 8.2
  - [ ] Mock gateway methods
  - [ ] Verify synced_facets, failed_facets, facet_miss_list populations

- [ ] **tests/test_intent_metadata.py**
  - [ ] Add tests for HydrationResult population
  - [ ] Verify backward compatibility with question_type

- [ ] **src/llm_stock_system/orchestrator/pipeline.py**
  - [ ] Update to capture `HydrationResult` from `hydrator.hydrate()`
  - [ ] Pass `facet_miss_list` to Validation Layer

- [ ] **src/llm_stock_system/layers/presentation_layer.py**
  - [ ] Update `query_logs` insertion to include `facet_miss_list`

- [ ] **docs/phase2-hydrator-refactoring.md** (this file)
  - [ ] Mark "Status: In Progress" → "Completed"
  - [ ] Add "Actual Effort" and "Date Completed" sections

---

## 10. Migration Path & Rollback

### 10.1 Rollout Strategy (3-Stage)

**Stage 1: Parallel Implementation** (Days 1-2)
- Implement new facet-based hydrate() alongside old question_type logic
- Add feature flag: `USE_FACET_BASED_HYDRATION = False`
- Tests pass (new tests + backward compat tests)

**Stage 2: Validation** (Day 2-3)
- Run against 10,000 sample queries
- Compare old vs new facet_miss_list, synced_facets
- Verify no data syncing regressions

**Stage 3: Cutover** (Day 3)
- Flip flag: `USE_FACET_BASED_HYDRATION = True`
- Monitor query_logs.facet_miss_list for unexpected patterns
- Retire old question_type detection code after 2 weeks

### 10.2 Rollback Plan

If issues detected within 48 hours:
1. Revert `USE_FACET_BASED_HYDRATION = False`
2. Queries will still populate intent/required_facets (Phase 0/1), but hydrator uses question_type
3. No data loss; no upstream impact

---

## 11. Success Criteria & Acceptance Tests

### Phase 2 Done Checklist

- [ ] All 24 tests pass
- [ ] Backward compatibility tests pass (question_type → intent auto-mapping)
- [ ] Code coverage ≥90% for query_data_hydrator.py
- [ ] Sample queries produce same facet_miss_list as manual verification
- [ ] No change to API contracts (pipeline.py still calls hydrate(query))
- [ ] Docstrings updated; intent/facet routing documented
- [ ] query_logs.facet_miss_list populated correctly in pipeline
- [ ] No upstream or downstream layer changes required
- [ ] GitHub PR reviewed and merged

### Performance Targets

- [ ] Hydration duration ≤ 5s per query (same as before)
- [ ] No additional DB calls beyond original design
- [ ] Memory usage unchanged (no new large data structures)

---

## 12. Risk Assessment

### Known Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gateway method signature mismatch | Low | High | Review all sync_* methods in gateway; add type hints |
| Missing facet → confidence scoring breaks | Medium | Medium | Validation Layer (Phase 3) handles facet_miss_list; hydrator just reports |
| Question type → Intent mapping incomplete | Low | Medium | QUESTION_TYPE_TO_INTENT already complete; just route instead of branch |
| Backward compat broken | Low | High | 2 compat tests + feature flag rollout |
| Date range logic differs from old code | Medium | Medium | Extract old date logic, compare per-facet results |

### Testing Assumptions

- Gateway methods are idempotent (can be called multiple times)
- Error handling (sync failures) doesn't crash hydrator
- Date range computations are deterministic

---

## 13. Future Phases (Post Phase 2)

### Phase 3: Validation Layer Enhancement
- Use `facet_miss_list` to adjust confidence_light
- Rule: missing required facet → RED or YELLOW light
- Track in `validation_result.warnings`

### Phase 4: Generation Layer Refactoring
- Replace 28 question_type branches in llm.py with 7 intent templates
- Use Intent + topic_tags + LLM for synthesis

### Phase 5: Generation Layer Redesign
- Move from RuleBasedSynthesis to LLM-driven generation
- Intent + topicTags + summary → GPT-4 → QueryResponse

---

## 14. Appendix: Code Examples

### Before: question_type branching (28 branches)

```python
def _hydrate_ticker(self, query: StructuredQuery, ticker: str, today: date) -> None:
    # ... time range setup ...
    
    if query.question_type in self.PRICE_RELATED_QUESTION_TYPES:
        self._safe_call(self._gateway.sync_price_history, ticker, price_start, today)
    
    if query.question_type == "technical_indicator_review":
        self._safe_call(self._gateway.sync_price_history, ticker, today - timedelta(days=180), today)
    
    if query.question_type == "season_line_margin_review":
        self._safe_call(self._gateway.sync_margin_purchase_short_sale, ticker, today - timedelta(days=180), today)
    
    # ... 13 more if blocks ...
```

### After: facet-based routing (1 loop)

```python
def hydrate(self, query: StructuredQuery) -> HydrationResult:
    # ... validation ...
    
    facets_to_sync = query.required_facets | query.preferred_facets
    
    for ticker in tickers:
        for facet in facets_to_sync:
            is_required = facet in query.required_facets
            facet_result = self._sync_facet(facet, ticker, query, today)
            
            if facet_result.success:
                result.synced_facets.add(facet)
            else:
                result.failed_facets[facet] = str(facet_result.exception)
                if is_required:
                    result.facet_miss_list.append(facet.value)
    
    return result
```

**Benefit**: Adding a new question_type doesn't require modifying hydrator. Just map it to Intent + facets in models.py QUESTION_TYPE_TO_INTENT.

---

## 15. Sign-Off

**Document Version**: 1.0  
**Author**: Claude (Cowork Agent)  
**Date Created**: April 14, 2026  
**Last Updated**: April 14, 2026  
**Status**: Completed  

**Next Step**: Monitor `facet_miss_list` in downstream validation and query log reporting.
