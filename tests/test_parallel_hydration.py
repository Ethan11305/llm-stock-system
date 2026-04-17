"""test_parallel_hydration.py

測試 P1：QueryDataHydrator ThreadPoolExecutor 並行 facet 拉取。

驗證項目：
1. 並行模式下，所有 facet 仍然都被呼叫（與串行結果相同）
2. 並行執行時，多個 facet 確實同時在不同 thread 執行
3. 某個 facet 失敗時，其他 facet 仍然繼續執行（隔離失敗）
4. parallel_hydration_workers=1 退回串行行為，結果不變
5. HydrationResult 欄位填充正確（synced_facets / facet_miss_list）
"""
from __future__ import annotations

import threading
import time
import unittest
from datetime import datetime, timezone

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import QueryRequest, StructuredQuery
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


# ─────────────────────────────────────────────────────────────────────────────
# 測試用 Gateway
# ─────────────────────────────────────────────────────────────────────────────

class _SlowGateway:
    """會記錄呼叫順序和 thread id 的 fake gateway。

    每個 sync 方法故意 sleep 50ms，讓並行測試能偵測到真正的平行執行。
    """

    def __init__(self, fail_facets: set[str] | None = None, sleep_ms: float = 50) -> None:
        self._fail_facets = fail_facets or set()
        self._sleep_s = sleep_ms / 1000
        self.call_log: list[tuple[str, int]] = []   # (method_name, thread_id)
        self._lock = threading.Lock()

    def _record(self, name: str) -> None:
        with self._lock:
            self.call_log.append((name, threading.get_ident()))
        time.sleep(self._sleep_s)
        if name in self._fail_facets:
            raise RuntimeError(f"Simulated failure: {name}")

    def sync_stock_info(self, force: bool = False) -> int:
        self._record("sync_stock_info")
        return 0

    def sync_price_history(self, ticker, start, end) -> int:
        self._record("sync_price_history")
        return 5

    def sync_financial_statements(self, ticker, start, end) -> int:
        self._record("sync_financial_statements")
        return 5

    def sync_monthly_revenue_points(self, ticker) -> int:
        self._record("sync_monthly_revenue_points")
        return 5

    def sync_dividend_policies(self, ticker, start, end) -> int:
        self._record("sync_dividend_policies")
        return 5

    def sync_balance_sheet_items(self, ticker, start, end) -> int:
        self._record("sync_balance_sheet_items")
        return 5

    def sync_cash_flow_statements(self, ticker, start, end) -> int:
        self._record("sync_cash_flow_statements")
        return 5

    def sync_pe_valuation_points(self, ticker) -> int:
        self._record("sync_pe_valuation_points")
        return 5

    def sync_margin_purchase_short_sale(self, ticker, start, end) -> int:
        self._record("sync_margin_purchase_short_sale")
        return 5

    def sync_stock_news(self, ticker, start, end) -> int:
        self._record("sync_stock_news")
        return 5


def _make_query(
    ticker: str = "2330",
) -> StructuredQuery:
    """建立測試用 StructuredQuery（直接指定多個 facets，不依賴 InputLayer 解析）。

    直接構建 StructuredQuery 以確保涵蓋足夠多的 facet（用於並行測試）。
    """
    return StructuredQuery(
        ticker=ticker,
        user_query=f"{ticker} 財務健康與估值",
        intent=Intent.INVESTMENT_ASSESSMENT,
        required_facets=[
            DataFacet.FINANCIAL_STATEMENTS,
            DataFacet.PE_VALUATION,
        ],
        preferred_facets=[
            DataFacet.PRICE_HISTORY,
            DataFacet.DIVIDEND,
            DataFacet.NEWS,
            DataFacet.MONTHLY_REVENUE,
            DataFacet.BALANCE_SHEET,
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 測試案例
# ─────────────────────────────────────────────────────────────────────────────

class ParallelHydrationTest(unittest.TestCase):

    def test_all_facets_called_in_parallel_mode(self):
        """並行模式下，所有 facet sync 方法都必須被呼叫。"""
        gateway = _SlowGateway(sleep_ms=10)
        hydrator = QueryDataHydrator(
            gateway,
            parallel_hydration_workers=6,
            low_confidence_warmup_enabled=False,
        )
        query = _make_query()
        result = hydrator.hydrate(query)

        called_methods = {name for name, _ in gateway.call_log}

        # _make_query() 的 required_facets 對應 sync 方法都必須出現
        # FINANCIAL_STATEMENTS → sync_financial_statements
        # PE_VALUATION         → sync_pe_valuation_points
        required_syncs = {"sync_financial_statements", "sync_pe_valuation_points"}
        for sync in required_syncs:
            self.assertIn(sync, called_methods, f"{sync} 未被呼叫")

        # preferred_facets 對應的 sync 也應被呼叫
        # PRICE_HISTORY → sync_price_history
        # NEWS → sync_stock_news
        preferred_syncs = {"sync_price_history", "sync_stock_news"}
        for sync in preferred_syncs:
            self.assertIn(sync, called_methods, f"preferred {sync} 未被呼叫")

        # synced_facets 不應為空
        self.assertTrue(len(result.synced_facets) > 0, "synced_facets 不應為空")

    def test_parallel_uses_multiple_threads(self):
        """並行模式下，facet 應跑在多個不同的 thread 上。"""
        gateway = _SlowGateway(sleep_ms=30)
        hydrator = QueryDataHydrator(
            gateway,
            parallel_hydration_workers=6,
            low_confidence_warmup_enabled=False,
        )
        query = _make_query()
        hydrator.hydrate(query)

        # 排除 sync_stock_info（它在主 thread 執行）
        facet_threads = {
            tid for method, tid in gateway.call_log
            if method != "sync_stock_info"
        }
        # 至少有 2 個不同 thread 同時執行（代表真正並行）
        self.assertGreater(len(facet_threads), 1, "並行模式應在多個 thread 中執行 facet")

    def test_serial_fallback_same_result(self):
        """workers=1 退回串行時，結果應與並行模式一致。"""
        query = _make_query()

        gateway_parallel = _SlowGateway(sleep_ms=5)
        parallel_result = QueryDataHydrator(
            gateway_parallel,
            parallel_hydration_workers=6,
            low_confidence_warmup_enabled=False,
        ).hydrate(query)

        gateway_serial = _SlowGateway(sleep_ms=5)
        serial_result = QueryDataHydrator(
            gateway_serial,
            parallel_hydration_workers=1,
            low_confidence_warmup_enabled=False,
        ).hydrate(query)

        # synced_facets 集合應相同
        self.assertEqual(
            parallel_result.synced_facets,
            serial_result.synced_facets,
            "串行/並行的 synced_facets 應一致",
        )
        # facet_miss_list 應相同
        self.assertEqual(
            set(parallel_result.facet_miss_list),
            set(serial_result.facet_miss_list),
            "串行/並行的 facet_miss_list 應一致",
        )

    def test_failed_facet_isolated_others_still_succeed(self):
        """某個 facet 失敗時，其他 facet 應繼續執行並成功。"""
        gateway = _SlowGateway(
            fail_facets={"sync_financial_statements"},
            sleep_ms=10,
        )
        hydrator = QueryDataHydrator(
            gateway,
            parallel_hydration_workers=6,
            low_confidence_warmup_enabled=False,
        )
        query = _make_query()
        result = hydrator.hydrate(query)

        # sync_financial_statements 失敗 → FINANCIAL_STATEMENTS 不在 synced_facets
        self.assertNotIn(DataFacet.FINANCIAL_STATEMENTS, result.synced_facets)

        # 其他 facet 應仍然成功（至少有一個在 synced_facets）
        other_facets = result.synced_facets - {DataFacet.FINANCIAL_STATEMENTS}
        self.assertTrue(len(other_facets) > 0, "其他 facet 應仍成功")

    def test_empty_tickers_returns_empty_result(self):
        """query.ticker=None 時應立即回傳空 HydrationResult。"""
        gateway = _SlowGateway()
        hydrator = QueryDataHydrator(gateway, low_confidence_warmup_enabled=False)
        query = StructuredQuery(
            ticker=None,
            user_query="市場分析",
            intent=Intent.NEWS_DIGEST,
        )
        result = hydrator.hydrate(query)

        self.assertEqual(len(result.synced_facets), 0)
        # gateway 的 sync 方法不應被呼叫（除了 sync_stock_info，它是在主 thread 先跑的）
        # 注意：sync_stock_info 由 _safe_call 保護，ticker=None 也會呼叫一次
        facet_calls = [m for m, _ in gateway.call_log if m != "sync_stock_info"]
        self.assertEqual(len(facet_calls), 0, "ticker=None 時不應呼叫任何 facet sync")

    def test_duration_ms_recorded(self):
        """result.total_duration_ms 應大於 0。"""
        gateway = _SlowGateway(sleep_ms=5)
        hydrator = QueryDataHydrator(gateway, low_confidence_warmup_enabled=False)
        result = hydrator.hydrate(_make_query())
        self.assertGreater(result.total_duration_ms, 0)

    def test_parallel_faster_than_serial(self):
        """並行執行應明顯比串行快（同樣的 facet，並行應 < 串行的 80%）。

        _make_query() 包含 7 個 facets，串行理論時間 = 7 × 100ms = 700ms，
        並行 6 workers 理論時間 ≈ 2 × 100ms = 200ms（明顯更快）。

        注意：此測試是「效能基準」，不是嚴格上限。
        在 CI 環境資源緊張時偶發失敗屬正常，可適當調整 threshold。
        """
        SLEEP_MS = 100   # 每個 facet sleep 100ms，確保並行差異明顯
        query = _make_query()

        # 計算任務數（避免 hard-code）
        gateway_probe = _SlowGateway(sleep_ms=0)
        hydrator_probe = QueryDataHydrator(gateway_probe, low_confidence_warmup_enabled=False)
        hydrator_probe.hydrate(query)
        n_facets = len([m for m, _ in gateway_probe.call_log if m != "sync_stock_info"])

        expected_serial_ms = n_facets * SLEEP_MS

        # 只在 n_facets >= 4 時才有意義進行並行效能測試
        if n_facets < 4:
            self.skipTest(f"facet 數量不足（{n_facets}），跳過並行效能測試")

        # 並行
        gateway_p = _SlowGateway(sleep_ms=SLEEP_MS)
        start = time.perf_counter()
        QueryDataHydrator(
            gateway_p,
            parallel_hydration_workers=6,
            low_confidence_warmup_enabled=False,
        ).hydrate(query)
        parallel_ms = (time.perf_counter() - start) * 1000

        # 並行應該明顯快於串行理論時間（< 70%，允許一定 thread 開銷）
        self.assertLess(
            parallel_ms,
            expected_serial_ms * 0.70,
            f"並行應比串行快（parallel={parallel_ms:.0f}ms，"
            f"serial_expected={expected_serial_ms:.0f}ms，"
            f"n_facets={n_facets}）",
        )


if __name__ == "__main__":
    unittest.main()
