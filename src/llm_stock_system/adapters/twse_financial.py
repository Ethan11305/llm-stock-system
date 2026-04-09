from datetime import datetime, timezone

import httpx

from llm_stock_system.core.models import MonthlyRevenuePoint, ValuationPoint


class TwseCompanyFinancialClient:
    """Official TWSE company financial and open data adapter."""

    def __init__(self, base_url: str, monthly_revenue_url: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._monthly_revenue_url = (
            monthly_revenue_url.rstrip("/")
            if monthly_revenue_url
            else "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
        )

    def fetch_monthly_revenue(self, ticker: str) -> list[MonthlyRevenuePoint]:
        rows = self._request_json(self._monthly_revenue_url)
        if not isinstance(rows, list):
            raise ValueError("Unexpected TWSE monthly revenue payload")

        matched_row = next(
            (row for row in rows if isinstance(row, dict) and str(row.get("公司代號", "")).strip() == ticker),
            None,
        )
        if matched_row is None:
            return []

        return [
            MonthlyRevenuePoint(
                ticker=ticker,
                revenue_month=self._parse_roc_month(str(matched_row.get("資料年月", ""))),
                revenue=self._coerce_float(matched_row.get("營業收入-當月營收")) or 0.0,
                prior_year_month_revenue=self._coerce_float(matched_row.get("營業收入-去年當月營收")),
                month_over_month_pct=self._coerce_float(matched_row.get("營業收入-上月比較增減(%)")),
                year_over_year_pct=self._coerce_float(matched_row.get("營業收入-去年同月增減(%)")),
                cumulative_revenue=self._coerce_float(matched_row.get("累計營業收入-當月累計營收")),
                prior_year_cumulative_revenue=self._coerce_float(matched_row.get("累計營業收入-去年累計營收")),
                cumulative_yoy_pct=self._coerce_float(matched_row.get("累計營業收入-前期比較增減(%)")),
                report_date=self._parse_roc_date(str(matched_row.get("出表日期", ""))),
                notes=self._normalize_text(matched_row.get("備註")),
            )
        ]

    def fetch_valuation_points(self, ticker: str) -> list[ValuationPoint]:
        payload = self._request_company_financial(ticker)
        chart = payload.get("chart", {})
        pe_section = chart.get("pe", {})
        pb_section = chart.get("pb", {})

        categories = pe_section.get("categories") or pb_section.get("categories") or []
        pe_series = pe_section.get("series", [])
        pb_series = pb_section.get("series", [])

        pe_values = pe_series[0].get("data", []) if len(pe_series) >= 1 else []
        peer_pe_values = pe_series[1].get("data", []) if len(pe_series) >= 2 else []
        pb_values = pb_series[0].get("data", []) if len(pb_series) >= 1 else []
        peer_pb_values = pb_section.get("series", [None, None])[1].get("data", []) if len(pb_series) >= 2 else []

        items: list[ValuationPoint] = []
        for index, category in enumerate(categories):
            items.append(
                ValuationPoint(
                    ticker=ticker,
                    valuation_month=self._parse_month(str(category)),
                    pe_ratio=self._coerce_float(pe_values[index] if index < len(pe_values) else None),
                    peer_pe_ratio=self._coerce_float(peer_pe_values[index] if index < len(peer_pe_values) else None),
                    pb_ratio=self._coerce_float(pb_values[index] if index < len(pb_values) else None),
                    peer_pb_ratio=self._coerce_float(peer_pb_values[index] if index < len(peer_pb_values) else None),
                )
            )
        return items

    def _request_company_financial(self, ticker: str) -> dict:
        payload = self._request_json(
            self._base_url,
            params={"code": ticker},
        )
        if not isinstance(payload, dict):
            raise ValueError("Unexpected TWSE company financial payload")
        return payload

    def _request_json(self, url: str, params: dict | None = None):
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    def _parse_month(self, raw_value: str) -> datetime:
        parsed = datetime.strptime(raw_value, "%Y%m")
        return datetime(parsed.year, parsed.month, 1, tzinfo=timezone.utc)

    def _parse_roc_month(self, raw_value: str) -> datetime:
        if len(raw_value) != 5 or not raw_value.isdigit():
            raise ValueError(f"Unexpected ROC month format: {raw_value}")
        year = int(raw_value[:3]) + 1911
        month = int(raw_value[3:])
        return datetime(year, month, 1, tzinfo=timezone.utc)

    def _parse_roc_date(self, raw_value: str) -> datetime | None:
        if len(raw_value) != 7 or not raw_value.isdigit():
            return None
        year = int(raw_value[:3]) + 1911
        month = int(raw_value[3:5])
        day = int(raw_value[5:])
        return datetime(year, month, day, tzinfo=timezone.utc)

    def _coerce_float(self, raw_value) -> float | None:
        if raw_value in (None, "", "None", "-"):
            return None
        return float(raw_value)

    def _normalize_text(self, raw_value) -> str | None:
        if raw_value in (None, "", "None", "-"):
            return None
        return str(raw_value).strip()
