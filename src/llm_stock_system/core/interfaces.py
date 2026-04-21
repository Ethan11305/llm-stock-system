from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from llm_stock_system.core.models import (
    AnswerDraft,
    Document,
    GovernanceReport,
    HydrationResult,
    PriceBar,
    QueryLogDetail,
    QueryResponse,
    StockInfo,
    SourceResponse,
    StructuredQuery,
    ValidationResult,
)

if TYPE_CHECKING:
    from llm_stock_system.adapters.vector_retrieval import SemanticSearchResult
    from llm_stock_system.services.embedding_service import EmbeddingResult


class DocumentRepository(Protocol):
    def upsert_documents(self, documents: list[Document]) -> int:
        ...

    def search_documents(self, query: StructuredQuery) -> list[Document]:
        ...


class QueryLogStore(Protocol):
    def save(
        self,
        query: StructuredQuery,
        response: QueryResponse,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> str:
        ...

    def get_sources(self, query_id: str) -> SourceResponse | None:
        ...

    def get_query_log(self, query_id: str) -> QueryLogDetail | None:
        ...


class QueryHydrator(Protocol):
    def hydrate(self, query: StructuredQuery) -> HydrationResult:
        ...


class StockResolver(Protocol):
    def resolve(self, query_text: str) -> tuple[str | None, str | None]:
        ...


class LLMClient(Protocol):
    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
    ) -> AnswerDraft:
        ...


class QueryClassifier(Protocol):
    """LLM 驅動的查詢分類器。

    負責將使用者自由文字轉為語意欄位，讓 InputLayer 擺脫上百行關鍵字比對。
    失敗時必須回傳 None，由 InputLayer 降級成 rule-based 判斷。

    回傳 dict 應包含（每個欄位皆可為 None／缺省，由 InputLayer 逐欄驗證）：
      - intent:                Intent 字串，例如 "valuation_check"
      - topic_tags:             list[str]，每個必須是 TopicTag enum 的中文值
      - time_range_label:       "1d"|"7d"|"30d"|"latest_quarter"|"1y"|"3y"|"5y"
      - stance_bias:            "bullish"|"bearish"|"neutral"

    Wave 4 Stage 5：分類器不再輸出 question_type；若需要 legacy 標籤，
    由 InputLayer 內部依 intent + controlled_tags 推導。
    """

    def classify(self, query_text: str) -> dict | None:
        ...


class PromptLoader(Protocol):
    def load(self, path: Path) -> str:
        ...


class MarketDataGateway(Protocol):
    def sync_stock_info(self, force: bool = False) -> int:
        ...

    def resolve_company(self, query_text: str) -> tuple[str | None, str | None]:
        ...

    def sync_price_history(self, ticker: str, start_date, end_date) -> int:
        ...

    def get_price_bars(self, ticker: str, start_date, end_date) -> list[PriceBar]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# P0 Embedding Pipeline Protocols