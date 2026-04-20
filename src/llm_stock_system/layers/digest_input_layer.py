"""Digest 產品線的 InputLayer 包裝。

職責：
    * 對外仍是 InputLayer 的 ``parse`` 契約（吃 QueryRequest，吐 StructuredQuery）
    * 固定把 ``StructuredQuery.query_profile`` 設為 ``SINGLE_STOCK_DIGEST``
    * 若 caller 沒帶 ``time_range``，強制使用 digest 的預設值（通常 7d）
    * Classifier 的「主+fallback」策略直接沿用 InputLayer._classify_semantics，
      不在此重複實作，避免兩套 rule 漂移

這個 layer 刻意「薄」——僅做 product-line 旗標與時間窗預設，不重寫語意判讀。
若未來需要 digest 專屬的 classifier prompt / schema，可再新增 classifier
adapter 並於 app.py 注入不同實例，**不須**改這一層。
"""

from __future__ import annotations

from llm_stock_system.core.enums import QueryProfile
from llm_stock_system.core.interfaces import QueryClassifier, StockResolver
from llm_stock_system.core.models import QueryRequest, StructuredQuery
from llm_stock_system.layers.input_layer import InputLayer


class DigestInputLayer:
    """Single Stock Digest 產品線專用的 InputLayer。

    Args:
        stock_resolver: 可選的 ticker/公司名解析器（與 InputLayer 同）
        classifier:      QueryClassifier 主路徑；None 時走純規則
        default_time_range_label: 沒帶 time_range 時的預設標籤
    """

    def __init__(
        self,
        stock_resolver: StockResolver | None = None,
        classifier: QueryClassifier | None = None,
        default_time_range_label: str = "7d",
    ) -> None:
        self._inner = InputLayer(stock_resolver=stock_resolver, classifier=classifier)
        self._default_time_range_label = default_time_range_label

    def parse(self, request: QueryRequest) -> StructuredQuery:
        # 1) Digest 的產品定義：7d 時間窗。若 caller 沒指定，就補預設。
        if not request.time_range:
            request = request.model_copy(update={"time_range": self._default_time_range_label})

        # 2) 真正的語意判讀交給 InputLayer（classifier 主 + per-field rule fallback）
        structured = self._inner.parse(request)

        # 3) 打上 digest 產品線旗標。此旗標會進 query_log，讓下游能追溯走的是 digest 路徑。
        return structured.model_copy(
            update={"query_profile": QueryProfile.SINGLE_STOCK_DIGEST}
        )


__all__ = ["DigestInputLayer"]
