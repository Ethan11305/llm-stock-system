from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from llm_stock_system.core.models import (
    AnswerDraft,
    Document,
    GovernanceReport,
    HydrationResult,
    PriceBar,
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
      - question_type:         QUESTION_TYPE_TO_INTENT 的合法 key
      - topic_tags:             list[str]，每個必須是 TopicTag enum 的中文值
      - time_range_label:       "1d"|"7d"|"30d"|"latest_quarter"|"1y"|"3y"|"5y"
      - stance_bias:            "bullish"|"bearish"|"neutral"
      - is_forecast_query:      bool
      - wants_direction:        bool
      - wants_scenario_range:   bool
      - forecast_horizon_label: str | None
      - forecast_horizon_days:  int | None
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
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingClient(Protocol):
    """Embedding 生成介面，方便未來替換模型（OpenAI / 本地模型等）。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """將文字列表轉為向量列表（維度依模型而異）。"""
        ...


class VectorRepository(Protocol):
    """向量檢索與儲存介面。"""

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SemanticSearchResult]:
        """以 cosine similarity 找最相似的 chunk。"""
        ...

    def upsert_embeddings(self, results: list[EmbeddingResult]) -> int:
        """批次寫入或更新 embedding。回傳實際寫入筆數。"""
        ...
