"""retrieval_layer.py

提供兩種檢索策略：

1. RetrievalLayer（原有）
   - 純 metadata 檢索（依 retrieval profile + builder plan 排序）
   - 無語意能力，但永遠有結果，作為 fallback 基底

2. HybridRetrievalLayer（P0 新增）
   - 兩路並行：metadata 檢索 + pgvector 語意檢索
   - 結果合併去重，混合排序（語意分數 × 0.6 + metadata 排名分數 × 0.4）
   - VectorRetrievalAdapter 不可用時，graceful fallback 到純 metadata
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_stock_system.core.interfaces import DocumentRepository
from llm_stock_system.core.models import Document, StructuredQuery

if TYPE_CHECKING:
    from llm_stock_system.adapters.vector_retrieval import (
        SemanticSearchResult,
        VectorRetrievalAdapter,
    )

logger = logging.getLogger(__name__)


class RetrievalLayer:
    """原有的純 metadata 檢索層（保持不變）。"""

    def __init__(self, document_repository: DocumentRepository, max_documents: int = 8) -> None:
        self._document_repository = document_repository
        self._max_documents = max_documents

    def retrieve(self, query: StructuredQuery) -> list[Document]:
        documents = self._document_repository.search_documents(query)
        return documents[: self._max_documents]


class HybridRetrievalLayer:
    """混合檢索層：結合 metadata 檢索 + pgvector 語意檢索。

    策略：
    1. 兩路並行（邏輯上並行，目前同步執行；P1 async 優化後可真正並行）
       - Path A：metadata 檢索（現有 builder 流程，保證永遠有結果）
       - Path B：語意檢索（pgvector cosine similarity）
    2. 結果合併去重（同一 document_id 只保留一份）
    3. 混合排序：語意分數 × semantic_weight + metadata 排名分數 × metadata_weight
    4. Vector adapter 不可用或失敗時，graceful fallback 到純 metadata
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_adapter: VectorRetrievalAdapter | None = None,
        max_documents: int = 8,
        semantic_weight: float = 0.6,
        metadata_weight: float = 0.4,
    ) -> None:
        self._document_repository = document_repository
        self._vector_adapter = vector_adapter
        self._max_documents = max_documents
        self._semantic_weight = semantic_weight
        self._metadata_weight = metadata_weight

    def retrieve(self, query: StructuredQuery) -> list[Document]:
        """混合檢索主方法。"""
        # Path A: 現有 metadata 檢索（永遠執行）
        metadata_docs = self._document_repository.search_documents(query)

        # 無 vector adapter → 直接 fallback
        if self._vector_adapter is None:
            return metadata_docs[: self._max_documents]

        # Path B: 語意檢索
        try:
            semantic_results: list[SemanticSearchResult] = self._vector_adapter.search(
                query_text=query.user_query,
                top_k=self._max_documents * 2,  # 多拉些，合併後去重
                ticker=query.ticker,
                days_back=query.time_range_days,
            )
        except Exception as exc:
            logger.warning("HybridRetrievalLayer: 語意檢索失敗，fallback 到純 metadata。原因：%s", exc)
            return metadata_docs[: self._max_documents]

        # 若語意檢索無結果，直接用 metadata 結果
        if not semantic_results:
            return metadata_docs[: self._max_documents]

        return self._merge_results(metadata_docs, semantic_results)

    # ──────────────────────────────────────
    # 內部合併邏輯
    # ──────────────────────────────────────

    def _merge_results(
        self,
        metadata_docs: list[Document],
        semantic_results: list[SemanticSearchResult],
    ) -> list[Document]:
        """合併兩路結果，計算混合分數後排序。"""
        # document_id → semantic similarity score
        semantic_scores: dict[str, float] = {
            r.document_id: r.similarity_score for r in semantic_results
        }

        # metadata_docs 的排名分數：第 1 名 = 1.0，依序遞減
        metadata_scores: dict[str, float] = {
            doc.id: max(0.0, 1.0 - i / max(len(metadata_docs), 1))
            for i, doc in enumerate(metadata_docs)
        }

        # document_id → Document 映射（從 metadata_docs 建立）
        doc_map: dict[str, Document] = {doc.id: doc for doc in metadata_docs}

        # 對只出現在 semantic_results 但不在 metadata_docs 的文件，補查完整 Document
        missing_ids = [
            did for did in semantic_scores if did not in doc_map
        ]
        if missing_ids:
            supplementary = self._fetch_documents_by_ids(missing_ids)
            doc_map.update({doc.id: doc for doc in supplementary})

        # 計算所有候選的混合分數
        all_doc_ids = set(metadata_scores.keys()) | set(semantic_scores.keys())
        scored: list[tuple[str, float]] = []
        for doc_id in all_doc_ids:
            if doc_id not in doc_map:
                continue
            sem = semantic_scores.get(doc_id, 0.0)
            meta = metadata_scores.get(doc_id, 0.0)
            hybrid = sem * self._semantic_weight + meta * self._metadata_weight
            scored.append((doc_id, hybrid))

        # 依混合分數降序
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc_map[doc_id] for doc_id, _ in scored[: self._max_documents]]

    def _fetch_documents_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """從 repository 補查語意檢索中出現但 metadata 流程沒拉到的文件。

        目前的 DocumentRepository 介面沒有 get_by_ids 方法，
        所以這裡用空 query 模式嘗試，若 repository 不支援則靜默忽略。
        """
        fetch_fn = getattr(self._document_repository, "get_documents_by_ids", None)
        if callable(fetch_fn):
            try:
                return fetch_fn(doc_ids)
            except Exception as exc:
                logger.warning("_fetch_documents_by_ids 失敗，忽略 %d 個文件：%s", len(doc_ids), exc)
        return []
