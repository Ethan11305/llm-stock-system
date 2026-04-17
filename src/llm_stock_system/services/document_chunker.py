"""document_chunker.py

將 Document 物件切成適合 embedding 的 chunk。

切片策略依文件長度而異：
- 短文件（< max_chunk_tokens）：整篇作為一個 chunk
- 長文件：按段落切，帶重疊，確保語意連貫性

中文估算：1 字 ≈ 0.5 token（粗估）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_stock_system.core.models import Document


@dataclass
class DocumentChunk:
    """文件切片，作為 embedding 的最小單位。"""

    document_id: str
    chunk_index: int
    chunk_text: str
    metadata: dict = field(default_factory=dict)
    # metadata 包含 ticker, source_tier, source_name, published_at, topics
    # 用於後續 vector 查詢時的 metadata 過濾


class DocumentChunker:
    """將 Document 切成適合 embedding 的 chunk。

    切片策略依文件類型而異：
    - 新聞文章：按段落切，每段 200-500 tokens
    - 財報數據：整份作為一個 chunk（已是結構化摘要）
    - 法說會逐字稿：按發言段落切，保留講者資訊
    """

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 30,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # ──────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """將單一 Document 切成多個 chunk。"""
        # 標題 + 內文拼接
        content = f"{document.title}\n\n{document.content}"
        # 中文約 1 字 ≈ 0.5 token（粗估）
        estimated_tokens = len(content) // 2

        metadata = self._extract_metadata(document)

        # 短文件：整篇作為一個 chunk
        if estimated_tokens <= self.max_chunk_tokens:
            return [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=0,
                    chunk_text=content.strip(),
                    metadata=metadata,
                )
            ]

        # 長文件：按段落切，帶重疊
        return self._chunk_by_paragraphs(document.id, content, metadata)

    def chunk_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """批次處理多個 Document。"""
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ──────────────────────────────────────
    # 內部方法
    # ──────────────────────────────────────

    def _chunk_by_paragraphs(
        self,
        document_id: str,
        content: str,
        metadata: dict,
    ) -> list[DocumentChunk]:
        """按段落切分，帶重疊滑窗。"""
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        chunks: list[DocumentChunk] = []
        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            candidate = f"{current_text}\n{para}" if current_text else para
            candidate_tokens = len(candidate) // 2

            if candidate_tokens > self.max_chunk_tokens and current_text:
                # 儲存當前 chunk
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        chunk_text=current_text.strip(),
                        metadata=metadata,
                    )
                )
                chunk_index += 1

                # 帶重疊：取 current_text 最後 overlap_tokens 字元
                overlap_chars = self.overlap_tokens * 2
                overlap_text = (
                    current_text[-overlap_chars:]
                    if len(current_text) > overlap_chars
                    else ""
                )
                current_text = f"{overlap_text}\n{para}" if overlap_text else para
            else:
                current_text = candidate

        # 最後一段
        if current_text and len(current_text) // 2 >= self.min_chunk_tokens:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    chunk_text=current_text.strip(),
                    metadata=metadata,
                )
            )

        # 若段落切分結果為空（罕見），fallback 整篇一個 chunk
        if not chunks:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=0,
                    chunk_text=content.strip(),
                    metadata=metadata,
                )
            )

        return chunks

    def _extract_metadata(self, document: Document) -> dict:
        return {
            "ticker": document.ticker,
            "source_tier": document.source_tier.value,
            "source_name": document.source_name,
            "published_at": document.published_at.isoformat(),
            "topics": [t.value for t in document.topics],
        }
