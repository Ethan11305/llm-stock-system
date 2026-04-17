import unittest

from llm_stock_system.core.enums import TopicTag
from llm_stock_system.core.models import QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class TopicTagCoverageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.input_layer = InputLayer()

    def test_shipping_query_hits_shipping_topic_tags(self) -> None:
        query = self.input_layer.parse(
            QueryRequest(query="長榮 2603 因紅海危機與 SCFI 運價走升，後續怎麼看？")
        )

        self.assertIn(TopicTag.SHIPPING, query.controlled_tags)
        self.assertIn("SCFI", query.free_keywords)

    def test_semiconductor_equipment_query_hits_semicon_equip_tag(self) -> None:
        query = self.input_layer.parse(
            QueryRequest(query="ASML 訂單不如預期，對家登 3680 和萬潤 6187 的半導體設備族群有何影響？")
        )

        self.assertIn(TopicTag.SEMICON_EQUIP, query.controlled_tags)
        self.assertIn("ASML", query.free_keywords)

    def test_comparison_query_keeps_fallback_tag_alongside_matched_tags(self) -> None:
        query = self.input_layer.parse(
            QueryRequest(query="我想比較長榮 (2603) 跟陽明 (2609)，這兩家公司誰的毛利率比較高？這代表哪一家的經營效率比較好？")
        )

        self.assertIn(TopicTag.GROSS_MARGIN, query.controlled_tags)
        self.asse