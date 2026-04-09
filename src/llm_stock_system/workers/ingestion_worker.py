from dataclasses import dataclass
from datetime import date

from llm_stock_system.adapters.postgres_market_data import FinMindPostgresGateway


@dataclass
class IngestionResult:
    stock_info_rows: int = 0
    price_rows: int = 0
    financial_statement_rows: int = 0
    dividend_rows: int = 0
    news_rows: int = 0


class IngestionWorker:
    def __init__(self, gateway: FinMindPostgresGateway) -> None:
        self._gateway = gateway

    def run_once(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        *,
        refresh_stock_info: bool = False,
        include_price: bool = True,
        include_financials: bool = True,
        include_dividend: bool = True,
        include_news: bool = True,
    ) -> IngestionResult:
        result = IngestionResult()
        if refresh_stock_info:
            result.stock_info_rows = self._gateway.sync_stock_info(force=True)
        if include_price:
            result.price_rows = self._gateway.sync_price_history(ticker, start_date, end_date)
        if include_financials:
            result.financial_statement_rows = self._gateway.sync_financial_statements(ticker, start_date, end_date)
        if include_dividend:
            result.dividend_rows = self._gateway.sync_dividend_policies(ticker, start_date, end_date)
        if include_news:
            result.news_rows = self._gateway.sync_stock_news(ticker, start_date, end_date)
        return result
