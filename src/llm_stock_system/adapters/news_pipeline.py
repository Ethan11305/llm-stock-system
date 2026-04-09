from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import html
import re
from typing import Callable
from urllib.parse import quote_plus
from xml.etree import ElementTree

import httpx

from llm_stock_system.core.enums import SourceTier
from llm_stock_system.core.models import NewsArticle


def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = html.unescape(cleaned)
    collapsed = " ".join(cleaned.split())
    return collapsed or None


class BaseNewsProvider:
    provider_name = "base"

    def fetch_articles(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        raise NotImplementedError


@dataclass(slots=True)
class FinMindNewsProvider(BaseNewsProvider):
    finmind_client: object
    provider_name: str = "finmind"

    def fetch_articles(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        del company_name, search_terms
        articles = self.finmind_client.fetch_stock_news(ticker, start_date, end_date)
        return [
            article.model_copy(
                update={
                    "provider_name": self.provider_name,
                    "source_tier": article.source_tier,
                    "source_type": article.source_type,
                }
            )
            for article in articles
        ]


class GoogleNewsRssProvider(BaseNewsProvider):
    provider_name = "google_news_rss"

    def __init__(
        self,
        base_url: str = "https://news.google.com/rss/search",
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("?")
        self._client_factory = client_factory or (lambda: httpx.Client(timeout=20.0, follow_redirects=True))

    def fetch_articles(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        articles: dict[tuple[str, str], NewsArticle] = {}
        queries = self._build_queries(ticker, company_name, search_terms)

        for query in queries:
            try:
                fetched = self._fetch_query_feed(query)
            except Exception:
                continue

            for article in fetched:
                if not self._is_within_range(article.published_at, start_date, end_date):
                    continue
                if not self._is_relevant(article, ticker, company_name, search_terms):
                    continue
                key = (article.title.strip(), article.url.strip())
                articles[key] = article.model_copy(
                    update={
                        "ticker": ticker,
                        "provider_name": self.provider_name,
                        "source_tier": SourceTier.MEDIUM,
                        "source_type": "news_article",
                        "tags": list(search_terms),
                    }
                )

        return sorted(articles.values(), key=lambda item: item.published_at, reverse=True)

    def _build_queries(
        self,
        ticker: str,
        company_name: str | None,
        search_terms: tuple[str, ...],
    ) -> list[str]:
        compact_name = re.sub(r"\s+", "", company_name or "").strip()
        base_subject = compact_name or ticker
        queries: list[str] = [base_subject]

        if compact_name and ticker:
            queries.append(f"{compact_name} {ticker}")

        if search_terms:
            focus_terms = " ".join(term for term in search_terms[:4] if term)
            if focus_terms:
                queries.append(f"{base_subject} {focus_terms}")

        unique_queries: list[str] = []
        for query in queries:
            normalized = " ".join(query.split()).strip()
            if normalized and normalized not in unique_queries:
                unique_queries.append(normalized)
        return unique_queries[:3]

    def _fetch_query_feed(self, query: str) -> list[NewsArticle]:
        url = (
            f"{self._base_url}?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant"
        )
        with self._client_factory() as client:
            response = client.get(url, headers={"Accept": "application/rss+xml, application/xml, text/xml"})
            response.raise_for_status()
            return self._parse_feed(response.text)

    def _parse_feed(self, payload: str) -> list[NewsArticle]:
        root = ElementTree.fromstring(payload)
        articles: list[NewsArticle] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue

            description = _strip_html(item.findtext("description"))
            source_name = (item.findtext("source") or "").strip() or "Google News"
            published_at = self._parse_pub_date(item.findtext("pubDate"))
            articles.append(
                NewsArticle(
                    ticker="",
                    published_at=published_at,
                    title=title,
                    summary=description,
                    source_name=source_name,
                    url=link,
                    source_tier=SourceTier.MEDIUM,
                    source_type="news_article",
                    provider_name=self.provider_name,
                )
            )
        return articles

    def _parse_pub_date(self, raw_value: str | None) -> datetime:
        if not raw_value:
            return datetime.now(timezone.utc)
        try:
            parsed = parsedate_to_datetime(raw_value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _is_within_range(self, published_at: datetime, start_date: date, end_date: date) -> bool:
        current = published_at.astimezone(timezone.utc).date()
        return start_date <= current <= end_date

    def _is_relevant(
        self,
        article: NewsArticle,
        ticker: str,
        company_name: str | None,
        search_terms: tuple[str, ...],
    ) -> bool:
        haystack = _normalize_lookup_text(f"{article.title} {article.summary or ''}")
        company_tokens = {
            _normalize_lookup_text(ticker),
            _normalize_lookup_text(company_name or ""),
        }
        company_tokens = {token for token in company_tokens if token}
        has_company_match = not company_tokens or any(token in haystack for token in company_tokens)

        if not search_terms:
            return has_company_match

        thematic_hits = 0
        for term in search_terms:
            normalized_term = _normalize_lookup_text(term)
            if normalized_term and normalized_term in haystack:
                thematic_hits += 1

        if has_company_match:
            return thematic_hits > 0 or len(search_terms) <= 1

        macro_sector_terms = {"cpi", "通膨", "殖利率", "高殖利率", "金控", "金控股", "法人", "外資", "利率", "美債"}
        normalized_terms = {_normalize_lookup_text(term) for term in search_terms if term}
        if normalized_terms.intersection({_normalize_lookup_text(term) for term in macro_sector_terms}):
            return thematic_hits >= 2

        return False


class MultiSourceNewsPipeline:
    def __init__(self, providers: list[BaseNewsProvider] | None = None) -> None:
        self._providers = providers or []

    @property
    def provider_names(self) -> list[str]:
        return [provider.provider_name for provider in self._providers]

    def fetch_stock_news(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        deduped: dict[tuple[str, str], NewsArticle] = {}
        for provider in self._providers:
            try:
                articles = provider.fetch_articles(
                    ticker=ticker,
                    company_name=company_name,
                    start_date=start_date,
                    end_date=end_date,
                    search_terms=search_terms,
                )
            except Exception:
                continue

            for article in articles:
                key = (article.title.strip(), article.url.strip())
                current = deduped.get(key)
                if current is None or self._is_better_article(article, current):
                    deduped[key] = article

        return sorted(deduped.values(), key=lambda item: item.published_at, reverse=True)

    def _is_better_article(self, candidate: NewsArticle, current: NewsArticle) -> bool:
        if candidate.source_tier != current.source_tier:
            return self._tier_rank(candidate.source_tier) > self._tier_rank(current.source_tier)
        return candidate.published_at >= current.published_at

    def _tier_rank(self, tier: SourceTier) -> int:
        return {
            SourceTier.HIGH: 3,
            SourceTier.MEDIUM: 2,
            SourceTier.LOW: 1,
        }[tier]
