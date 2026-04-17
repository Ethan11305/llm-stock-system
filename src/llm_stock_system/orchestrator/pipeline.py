import logging

from llm_stock_system.core.interfaces import QueryHydrator, QueryLogStore
from llm_stock_system.core.models import HydrationResult, QueryRequest, QueryResponse, SourceResponse
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer

logger = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(
        self,
        input_layer: InputLayer,
        retrieval_layer: RetrievalLayer,
        data_governance_layer: DataGovernanceLayer,
        generation_layer: GenerationLayer,
        validation_layer: ValidationLayer,
        presentation_layer: PresentationLayer,
        query_log_store: QueryLogStore,
        query_hydrator: QueryHydrator | None = None,
    ) -> None:
        self._input_layer = input_layer
        self._retrieval_layer = retrieval_layer
        self._data_governance_layer = data_governance_layer
        self._generation_layer = generation_layer
        self._validation_layer = validation_layer
        self._presentation_layer = presentation_layer
        self._query_log_store = query_log_store
        self._query_hydrator = query_hydrator

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        structured_query = self._input_layer.parse(request)
        hydration_result = HydrationResult()
        if self._query_hydrator is not None:
            try:
                candidate_result = self._query_hydrator.hydrate(structured_query)
                if isinstance(candidate_result, HydrationResult):
                    hydration_result = candidate_result
            except Exception as exc:
                # 記錄完整 traceback，讓 thread-pool 帶來的失敗可被追蹤
                # 主流程繼續執行，但 hydration_result 保持為空（等同資料缺失）
                logger.error(
                    "QueryPipeline: hydration 失敗（ticker=%s, intent=%s）：%s",
                    getattr(structured_query, "ticker", None),
                    getattr(structured_query, "intent", None),
                    exc,
                    exc_info=True,
                )
        retrieved_documents = self._retrieval_layer.retrieve(structured_query)
        governance_report = self._data_governance_layer.curate(structured_query, retrieved_documents)
        answer_draft = self._generation_layer.generate(structured_query, governance_report)
        validation_result = self._validation_layer.validate(
            structured_query,
            governance_report,
            answer_draft,
            hydration_result.facet_miss_list,
            hydration_result.preferred_miss_list,
        )
        response = self._presentation_layer.present(
            answer_draft,
            governance_report,
            validation_result,
        )
        query_id = self._query_log_store.save(
            structured_query,
            response,
            governance_report,
            validation_result,
        )
        response.query_id = query_id
        follow_up_scheduler = getattr(self._query_hydrator, "schedule_follow_up", None)
        if callable(follow_up_scheduler):
            try:
                follow_up_scheduler(structured_query, validation_result)
            except Exception as exc:
                logger.warning(
                    "QueryPipeline: schedule_follow_up 失敗（ticker=%s）：%s",
                    getattr(structured_query, "ticker", None),
                    exc,
                    exc_info=True,
                )
        return response