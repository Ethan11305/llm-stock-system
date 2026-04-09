import argparse
from datetime import datetime, timedelta, timezone

from llm_stock_system.adapters.finmind import FinMindClient
from llm_stock_system.adapters.news_pipeline import (
    FinMindNewsProvider,
    GoogleNewsRssProvider,
    MultiSourceNewsPipeline,
)
from llm_stock_system.adapters.postgres_market_data import FinMindPostgresGateway
from llm_stock_system.adapters.twse_financial import TwseCompanyFinancialClient
from llm_stock_system.core.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync FinMind market data into PostgreSQL.")
    parser.add_argument("--ticker", help="Taiwan stock ticker to sync, e.g. 2330")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of price history to sync.",
    )
    parser.add_argument(
        "--price",
        action="store_true",
        help="Sync TaiwanStockPrice for the ticker.",
    )
    parser.add_argument(
        "--stock-info",
        action="store_true",
        help="Refresh TaiwanStockInfo before syncing prices.",
    )
    parser.add_argument(
        "--force-stock-info",
        action="store_true",
        help="Force refresh stock info even if cached data looks fresh.",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Sync TaiwanStockFinancialStatements and TaiwanStockCashFlowsStatement for the ticker.",
    )
    parser.add_argument(
        "--balance-sheet",
        action="store_true",
        help="Sync TaiwanStockBalanceSheet for the ticker.",
    )
    parser.add_argument(
        "--cash-flow",
        action="store_true",
        help="Sync TaiwanStockCashFlowsStatement for the ticker.",
    )
    parser.add_argument(
        "--monthly-revenue",
        action="store_true",
        help="Sync official TWSE monthly revenue history for the ticker.",
    )
    parser.add_argument(
        "--valuation",
        action="store_true",
        help="Sync official TWSE PE/PB valuation history for the ticker.",
    )
    parser.add_argument(
        "--dividend",
        action="store_true",
        help="Sync TaiwanStockDividend for the ticker.",
    )
    parser.add_argument(
        "--news",
        action="store_true",
        help="Sync TaiwanStockNews for the ticker.",
    )
    parser.add_argument(
        "--news-keyword",
        action="append",
        default=[],
        help="Optional news keyword to expand RSS/news search context. Repeatable.",
    )
    parser.add_argument(
        "--margin",
        action="store_true",
        help="Sync TaiwanStockMarginPurchaseShortSale for the ticker.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    finmind_client = FinMindClient(
        base_url=settings.finmind_base_url,
        api_token=settings.finmind_api_token,
    )
    twse_financial_client = TwseCompanyFinancialClient(
        base_url=settings.twse_company_financial_url,
        monthly_revenue_url=settings.twse_monthly_revenue_url,
    )
    news_pipeline = None
    if settings.news_pipeline_enabled:
        news_providers = [FinMindNewsProvider(finmind_client)]
        if settings.google_news_rss_enabled:
            news_providers.append(GoogleNewsRssProvider(base_url=settings.google_news_rss_base_url))
        news_pipeline = MultiSourceNewsPipeline(news_providers)
    gateway = FinMindPostgresGateway(
        database_url=settings.database_url,
        finmind_client=finmind_client,
        twse_financial_client=twse_financial_client,
        news_pipeline=news_pipeline,
        sync_on_query=settings.finmind_sync_on_query,
        stock_info_refresh_hours=settings.stock_info_refresh_hours,
    )

    if args.stock_info or args.force_stock_info:
        synced_count = gateway.sync_stock_info(force=args.force_stock_info)
        print(f"Synced stock info rows: {synced_count}")

    if args.ticker:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=args.days)
        requested_non_price_sync = any(
            (
                args.fundamentals,
                args.balance_sheet,
                args.cash_flow,
                args.monthly_revenue,
                args.valuation,
                args.dividend,
                args.news,
                args.margin,
            )
        )
        if args.price or not requested_non_price_sync:
            synced_bars = gateway.sync_price_history(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_bars} price bars for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.fundamentals:
            synced_items = gateway.sync_financial_statements(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_items} financial statement rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.fundamentals or args.balance_sheet:
            synced_balance_sheet_items = gateway.sync_balance_sheet_items(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_balance_sheet_items} balance sheet rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.fundamentals or args.cash_flow:
            synced_cash_flow_items = gateway.sync_cash_flow_statements(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_cash_flow_items} cash flow statement rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.dividend:
            synced_dividend = gateway.sync_dividend_policies(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_dividend} dividend rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.news:
            synced_news = gateway.sync_stock_news(
                args.ticker,
                start_date,
                end_date,
                search_terms=tuple(args.news_keyword),
            )
            print(
                f"Synced {synced_news} news rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.margin:
            synced_margin = gateway.sync_margin_purchase_short_sale(args.ticker, start_date, end_date)
            print(
                f"Synced {synced_margin} margin rows for {args.ticker} "
                f"from {start_date.isoformat()} to {end_date.isoformat()}"
            )
        if args.monthly_revenue:
            synced_monthly_revenue = gateway.sync_monthly_revenue_points(args.ticker)
            print(f"Synced {synced_monthly_revenue} monthly revenue rows for {args.ticker}")
        if args.valuation:
            synced_valuation = gateway.sync_pe_valuation_points(args.ticker)
            print(f"Synced {synced_valuation} valuation rows for {args.ticker}")

    if (
        not args.ticker
        and not args.stock_info
        and not args.force_stock_info
        and not args.price
        and not args.fundamentals
        and not args.balance_sheet
        and not args.cash_flow
        and not args.monthly_revenue
        and not args.valuation
        and not args.dividend
        and not args.news
        and not args.margin
    ):
        parser.error("Specify --ticker and/or --stock-info")


if __name__ == "__main__":
    main()
