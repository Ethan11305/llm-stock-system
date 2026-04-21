from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import FreshnessStatus, ConsistencyStatus, Intent, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def sync_stock_info(self, force: bool = False) -> int:
        self.calls.append(("sync_stock_info", (force,)))
        return 0

    def sync_monthly_revenue_points(self, ticker) -> int:
        self.calls.append(("sync_monthly_revenue_points", (ticker,)))
        return 0

    def sync_stock_news(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_stock_news", (ticker, start_date, end_date)))
        return 0


class MonthlyRevenueEventQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_monthly_revenue_event_query(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="剛公佈的「3035 智原」5 月營收，月增率 (MoM) 是否超過 20%？請搜尋這是否創下近一年新高，並摘要市場對此突發成長的解讀。"
            )
        )

        self.assertEqual(query.ticker, "3035")
        self.assertEqual(query.intent, Intent.EARNINGS_REVIEW)
        self.assertEqual(query.time_range_days, 365)

    def test_rule_based_summary_explains_future_month_not_published(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="剛公佈的「3035 智原」5 月營收，月增率 (MoM) 是否超過 20%？請搜尋這是否創下近一年新高，並摘要市場對此突發成長的解讀。"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="智原 2026-05 月營收尚未出表",
                    excerpt="截至 2026-04-08，官方月營收資料最新僅到 2026-02，尚未公布 2026-05 月營收。因此目前無法判定該月月增率是否超過 20%，也無法確認是否創下近一年新高。",
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_tier=SourceTier.HIGH,
                    url="https://openapi.twse.com.tw/v1/opendata/t187ap05_L#revenue-availability",
                    published_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="智原 最新已公布月營收摘要",
                    excerpt="智原 目前最新已公布月營收為 2026-02，單月營收約 7.37 億元。月增率約 -6.13%。年增率約 -74.02%。公司備註：因量產減少。",
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_tier=SourceTier.HIGH,
                    url="https://openapi.twse.com.tw/v1/opendata/t187ap05_L#latest-revenue",
                    published_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("截至 2026-04-08", draft.summary)
        self.assertIn("尚未公布 2026-05 月營收", draft.summary)

    def test_hydrator_fetches_monthly_revenue_and_news_for_event_query(self) -> None:
        gateway = FakeGateway()
        hydrator = QueryDataHydrator(gateway)
        query = InputLayer().parse(
            QueryRequest(
                query="剛公佈的「3035 智原」5 月營收，月增率 (MoM) 是否超過 20%？請搜尋這是否創下近一年新高，並摘要市場對此突發成長的解讀。"
            )
        )

        hydrator.hydrate(query)

        methods = [method_name for method_name, _ in gateway.calls]
        self.assertIn("sync_monthly_revenue_points", methods)
        self.assertIn("sync_stock_news", methods)


if __name__ == "__main__":
    unittest.main()
