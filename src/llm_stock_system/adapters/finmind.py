import time as _time
from datetime import date, datetime, time, timedelta, timezone

import httpx

from llm_stock_system.core.enums import SourceTier
from llm_stock_system.core.models import (
    DividendPolicy,
    FinancialStatementItem,
    MarginPurchaseShortSale,
    NewsArticle,
    PriceBar,
    StockInfo,
)


class FinMindRateLimitError(Exception):
    """FinMind API 回傳 429，免費方案已觸及 Rate Limit。

    Attributes:
        user_message: 可直接顯示給使用者的說明文字。
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class FinMindClient:
    """Thin adapter around the FinMind v4 data API."""

    def __init__(self, base_url: str, api_token: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token.strip()

    def fetch_stock_info(self) -> list[StockInfo]:
        payload = self._request_dataset("TaiwanStockInfo")
        stock_info: list[StockInfo] = []

        for row in payload:
            reference_date = row.get("date")
            stock_info.append(
                StockInfo(
                    stock_id=str(row["stock_id"]),
                    stock_name=str(row["stock_name"]),
                    industry_category=row.get("industry_category"),
                    market_type=row.get("type"),
                    reference_date=self._parse_date(reference_date)
                    if reference_date not in (None, "", "None")
                    else None,
                )
            )

        return stock_info

    def fetch_stock_price(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        payload = self._request_dataset(
            "TaiwanStockPrice",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        price_bars: list[PriceBar] = []

        for row in payload:
            trading_date = self._parse_date(row["date"])
            price_bars.append(
                PriceBar(
                    ticker=str(row["stock_id"]),
                    trading_date=trading_date,
                    open_price=float(row["open"]),
                    high_price=float(row["max"]),
                    low_price=float(row["min"]),
                    close_price=float(row["close"]),
                    trading_volume=self._as_optional_int(row.get("Trading_Volume")),
                    trading_money=self._as_optional_int(row.get("Trading_money")),
                    spread=self._as_optional_float(row.get("spread")),
                    turnover=self._as_optional_int(row.get("Trading_turnover")),
                )
            )

        return price_bars

    def fetch_financial_statements(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[FinancialStatementItem]:
        payload = self._request_dataset(
            "TaiwanStockFinancialStatements",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        statement_items: list[FinancialStatementItem] = []

        for row in payload:
            value = row.get("value")
            if value in (None, "", "None"):
                continue
            statement_items.append(
                FinancialStatementItem(
                    ticker=str(row["stock_id"]),
                    statement_date=self._parse_date(row["date"]),
                    item_type=str(row["type"]),
                    value=float(value),
                    origin_name=str(row.get("origin_name") or row["type"]),
                )
            )

        return statement_items

    def fetch_balance_sheet_items(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[FinancialStatementItem]:
        payload = self._request_dataset(
            "TaiwanStockBalanceSheet",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        statement_items: list[FinancialStatementItem] = []

        for row in payload:
            value = row.get("value")
            if value in (None, "", "None"):
                continue
            statement_items.append(
                FinancialStatementItem(
                    ticker=str(row["stock_id"]),
                    statement_date=self._parse_date(row["date"]),
                    item_type=str(row["type"]),
                    value=float(value),
                    origin_name=str(row.get("origin_name") or row["type"]),
                )
            )

        return statement_items

    def fetch_cash_flow_statements(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[FinancialStatementItem]:
        payload = self._request_dataset(
            "TaiwanStockCashFlowsStatement",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        statement_items: list[FinancialStatementItem] = []

        for row in payload:
            value = row.get("value")
            if value in (None, "", "None"):
                continue
            statement_items.append(
                FinancialStatementItem(
                    ticker=str(row["stock_id"]),
                    statement_date=self._parse_date(row["date"]),
                    item_type=str(row["type"]),
                    value=float(value),
                    origin_name=str(row.get("origin_name") or row["type"]),
                )
            )

        return statement_items

    def fetch_dividend_policies(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[DividendPolicy]:
        payload = self._request_dataset(
            "TaiwanStockDividend",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        policies: list[DividendPolicy] = []

        for row in payload:
            row_date = row.get("date")
            if row_date in (None, "", "None"):
                continue
            policies.append(
                DividendPolicy(
                    ticker=str(row["stock_id"]),
                    date=self._parse_date(row_date),
                    year_label=str(row.get("year") or ""),
                    cash_earnings_distribution=self._as_optional_float(row.get("CashEarningsDistribution")),
                    cash_statutory_surplus=self._as_optional_float(row.get("CashStatutorySurplus")),
                    stock_earnings_distribution=self._as_optional_float(row.get("StockEarningsDistribution")),
                    stock_statutory_surplus=self._as_optional_float(row.get("StockStatutorySurplus")),
                    participate_distribution_of_total_shares=self._as_optional_float(
                        row.get("ParticipateDistributionOfTotalShares")
                    ),
                    announcement_date=self._parse_date(row.get("AnnouncementDate"))
                    if row.get("AnnouncementDate") not in (None, "", "None")
                    else None,
                    announcement_time=row.get("AnnouncementTime"),
                    cash_ex_dividend_trading_date=self._parse_date(row.get("CashExDividendTradingDate"))
                    if row.get("CashExDividendTradingDate") not in (None, "", "None")
                    else None,
                    cash_dividend_payment_date=self._parse_date(row.get("CashDividendPaymentDate"))
                    if row.get("CashDividendPaymentDate") not in (None, "", "None")
                    else None,
                )
            )

        return policies

    def fetch_stock_news(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[NewsArticle]:
        news: dict[tuple[str, str], NewsArticle] = {}
        current = start_date

        while current <= end_date:
            try:
                payload = self._request_dataset(
                    "TaiwanStockNews",
                    data_id=ticker,
                    start_date=current.isoformat(),
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 402:
                    break
                raise

            for row in payload:
                link = str(row.get("link") or row.get("url") or "")
                title = str(row.get("title") or "").strip()
                if not link or not title:
                    continue

                article = NewsArticle(
                    ticker=str(row.get("stock_id") or ticker),
                    published_at=self._parse_datetime_value(row.get("date") or row.get("pub_date")),
                    title=title,
                    summary=row.get("summary") or row.get("content"),
                    source_name=str(row.get("source") or row.get("provider") or "FinMind TaiwanStockNews"),
                    url=link,
                    source_tier=SourceTier.MEDIUM,
                    source_type="news_article",
                    provider_name="finmind",
                )
                news[(article.title, article.url)] = article

            current += timedelta(days=1)

        return sorted(news.values(), key=lambda item: item.published_at, reverse=True)

    def fetch_margin_purchase_short_sale(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[MarginPurchaseShortSale]:
        payload = self._request_dataset(
            "TaiwanStockMarginPurchaseShortSale",
            data_id=ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        rows: list[MarginPurchaseShortSale] = []

        for row in payload:
            trading_date = row.get("date")
            if trading_date in (None, "", "None"):
                continue
            rows.append(
                MarginPurchaseShortSale(
                    ticker=str(row.get("stock_id") or ticker),
                    trading_date=self._parse_date(trading_date),
                    margin_purchase_buy=self._as_optional_int(row.get("MarginPurchaseBuy")),
                    margin_purchase_cash_repayment=self._as_optional_int(row.get("MarginPurchaseCashRepayment")),
                    margin_purchase_limit=self._as_optional_int(row.get("MarginPurchaseLimit")),
                    margin_purchase_sell=self._as_optional_int(row.get("MarginPurchaseSell")),
                    margin_purchase_today_balance=self._as_optional_int(row.get("MarginPurchaseTodayBalance")),
                    margin_purchase_yesterday_balance=self._as_optional_int(row.get("MarginPurchaseYesterdayBalance")),
                    offset_loan_and_short=self._as_optional_int(row.get("OffsetLoanAndShort")),
                    short_sale_buy=self._as_optional_int(row.get("ShortSaleBuy")),
                    short_sale_cash_repayment=self._as_optional_int(row.get("ShortSaleCashRepayment")),
                    short_sale_limit=self._as_optional_int(row.get("ShortSaleLimit")),
                    short_sale_sell=self._as_optional_int(row.get("ShortSaleSell")),
                    short_sale_today_balance=self._as_optional_int(row.get("ShortSaleTodayBalance")),
                    short_sale_yesterday_balance=self._as_optional_int(row.get("ShortSaleYesterdayBalance")),
                    note=row.get("note"),
                )
            )

        return rows

    def _request_dataset(self, dataset: str, **params) -> list[dict]:
        """呼叫 FinMind /data，帶指數退避 retry。

        重試條件：
        - 429 Too Many Requests（免費方案 rate limit）：最多重試 3 次（2s → 4s → 8s）
        - 5xx Server Error：同上
        - httpx.TimeoutException：同上

        429 重試耗盡後拋 FinMindRateLimitError（含可顯示給使用者的說明文字）。
        其他 4xx 直接往上拋，不重試。
        """
        query = {"dataset": dataset}
        query.update(params)

        last_exc: Exception | None = None
        is_rate_limited = False

        for attempt in range(3):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(
                        f"{self._base_url}/data",
                        headers=self._headers(),
                        params=query,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    data = payload.get("data", [])
                    if not isinstance(data, list):
                        raise ValueError(f"Unexpected FinMind payload for dataset {dataset}")
                    return data
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status == 429:
                    is_rate_limited = True
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    _time.sleep(wait)
                elif status >= 500:
                    wait = 2 ** (attempt + 1)
                    _time.sleep(wait)
                else:
                    raise
            except httpx.TimeoutException as exc:
                last_exc = exc
                _time.sleep(2 ** (attempt + 1))

        if is_rate_limited:
            raise FinMindRateLimitError(
                "FinMind API 已觸及免費方案的使用上限（Rate Limit）。"
                "目前市場資料無法即時補充，回答將以現有本地資料為準。"
                "若需完整資料，請稍後再試或考慮升級 FinMind 付費方案。"
            )
        raise last_exc or RuntimeError(f"FinMind dataset {dataset} 請求失敗（重試 3 次）")

    def _headers(self) -> dict[str, str]:
        if not self._api_token:
            return {}
        return {"Authorization": f"Bearer {self._api_token}"}

    def _parse_date(self, raw: str) -> datetime:
        parsed = datetime.strptime(raw, "%Y-%m-%d")
        return datetime.combine(parsed.date(), time.min, tzinfo=timezone.utc)

    def _parse_datetime_value(self, raw: str | None) -> datetime:
        if raw in (None, "", "None"):
            return datetime.now(timezone.utc)

        candidates = (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        )
        for pattern in candidates:
            try:
                parsed = datetime.strptime(raw, pattern)
                if pattern in ("%Y-%m-%d", "%Y/%m/%d"):
                    return datetime.combine(parsed.date(), time.min, tzinfo=timezone.utc)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return datetime.now(timezone.utc)

    def _as_optional_int(self, value) -> int | None:
        if value in (None, "", "None"):
            return None
        return int(float(value))

    def _as_optional_float(self, value) -> float | None:
        if value in (None, "", "None"):
            return None
        return float(value)
