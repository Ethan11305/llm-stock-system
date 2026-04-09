from llm_stock_system.core.interfaces import DocumentRepository
from llm_stock_system.core.models import Document, StructuredQuery


class RetrievalLayer:
    def __init__(self, document_repository: DocumentRepository, max_documents: int = 8) -> None:
        self._document_repository = document_repository
        self._max_documents = max_documents

    def retrieve(self, query: StructuredQuery) -> list[Document]:
        documents = self._document_repository.search_documents(query)
        return documents[: self._max_documents]
