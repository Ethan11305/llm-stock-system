"""augmentation_layer.py — RAG 的 A（Augmentation）層

職責：在 DataGovernanceLayer（R 結束）與 GenerationLayer（G 開始）之間，
把 GovernanceReport 裡的 evidence 重組成對 LLM 真正有用的上下文。

設計要點：
  - 每個 Intent 對應一個 AugmentationStrategy（adapters/augmentation/）
  - 策略從 evidence.source_type 識別結構化資料，格式化成 structured_block
  - 新聞 / 公告保留在 narrative_texts，讓 LLM 做語意推理
  - data_gaps 明確告知 LLM 缺什麼，避免幻覺補全
  - 若無對應策略（或 evidence 為空），回傳空 AugmentedContext，
    pipeline 仍可正常執行（向後相容）
"""
from __future__ import annotations

import logging

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery

logger = logging.getLogger(__name__)


class AugmentationLayer:
    """根據 query.intent 派發到對應的 AugmentationStrategy，產生 AugmentedContext。"""

    def augment(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
    ) -> AugmentedContext:
        if not governance_report.evidence:
            return AugmentedContext()

        from llm_stock_system.adapters.augmentation.registry import get_augmentation_strategy

        strategy = get_augmentation_strategy(query.intent)
        if strategy is None:
            logger.debug(
                "AugmentationLayer: intent=%s 無對應策略，回傳空 context",
                query.intent.value,
            )
            return AugmentedContext()

        try:
            return strategy.build(query, governance_report)
        except Exception as exc:
            logger.warning(
                "AugmentationLayer: strategy 執行失敗（intent=%s）：%s",
                query.intent.value,
                exc,
                exc_info=True,
            )
            return AugmentedContext()
