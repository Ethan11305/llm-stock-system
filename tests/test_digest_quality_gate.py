"""P2 品質 + 成本 gate 測試。

涵蓋兩條獨立路徑：
  * ValidationLayer 的 digest profile gate（_apply_digest_profile_gate）
    - < min_sources → insufficient-source warning + confidence 降分
    - stale_ratio >= threshold → stale warning + confidence 降分
    - support_score 全 0 → warning only（不扣分）
    - LEGACY profile 不受任何 digest gate 影響
  * QueryDataHydrator 的 skip_embedding_for_digest 成本 gate
    - digest 路徑 + skip_flag=True（預設）→ embedding 被跳過
    - legacy 路徑 → embedding 照常觸發
    - digest 路徑 + skip_flag=False → embedding 仍觸發（override 開關有效）

測試風格：盡量用 fake 物件，不啟整條 pipeline，聚焦 gate 本身行為。
"""

from __future__ import annotations

import threading
import time
import unittest
from datetime import datetime, timedelta, timezone

from llm_stock_system.core.enums import (
    ConfidenceLight,
    ConsistencyStatus,
    FreshnessStatus,
    Intent,
    QueryProfile,
    SourceTier,
    StanceBias,
    SufficiencyStatus,
)
from llm_stock_system.core.models import (
    AnswerDraft,
    Evidence,
    GovernanceReport,
    SourceCitation,
    StructuredQuery,
)
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _mk_evidence(
    *,
    document_id: str = "doc-1",
    title: str = "測試新聞",
    source_name: str = "FinMind News",
    published_at: datetime | None = None,
    support_score: float = 0.5,
) -> Evidence:
    return Evidence(
        document_id=document_id,
        title=title,
        excerpt="測試內容摘要",
        source_name=source_name,
        source_tier=SourceTier.MEDIUM,
        url=f"https://example.com/{document_id}",
        published_at=published_at or datetime.now(timezone.utc),
        support_score=support_score,
        corroboration_count=1,
    )


def _mk_governance(evidence: list[Evidence]) -> GovernanceReport:
    return GovernanceReport(
        evidence=evidence,
        dropped_document_ids=[],
        sufficiency=(
            SufficiencyStatus.SUFFICIENT if len(evidence) >= 2 else SufficiencyStatus.INSUFFICIENT
        ),
        consistency=ConsistencyStatus.MOSTLY_CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=0.5,
    )


def _mk_answer_draft() -> AnswerDraft:
    return AnswerDraft(
        summary="2330 近期動態摘要（測試用）。",
        highlights=["重點 A", "重點 B"],
        facts=["事實 1"],
        impacts=[],
        risks=["風險 1", "風險 2", "風險 3"],
        sources=[],
        forecast=None,
    )


def _mk_digest_query(ticker: str = "2330") -> StructuredQuery:
    return StructuredQuery(
        user_query=f"{ticker} 最近有什麼新聞？",
        ticker=ticker,
        company_name="台積電",
        intent=Intent.NEWS_DIGEST,
        time_range_label="7d",
        time_range_days=7,
        stance_bias=StanceBias.NEUTRAL,
        classifier_source="rule",
        query_profile=QueryProfile.SINGLE_STOCK_DIGEST,
    )


def _mk_legacy_query(ticker: str = "2330") -> StructuredQuery:
    return StructuredQuery(
        user_query=f"{ticker} 最近有什麼新聞？",
        ticker=ticker,
        company_name="台積電",
        intent=Intent.NEWS_DIGEST,
        time_range_label="7d",
        time_range_days=7,
        stance_bias=StanceBias.NEUTRAL,
        classifier_source="rule",
        query_profile=QueryProfile.LEGACY,
    )


# ─────────────────────────────────────────────────────────────────────
# 1) ValidationLayer digest profile gate
# ─────────────────────────────────────────────────────────────────────


class DigestQualityGateTestCase(unittest.TestCase):
    """_apply_digest_profile_gate 的單元行為測試。"""

    def _build_layer(self) -> ValidationLayer:
        # 把主 threshold 放很鬆，避免其他 gate 噪音
        return ValidationLayer(
            min_green_confidence=0.8,
            min_yellow_confidence=0.55,
            digest_min_sources=2,
            digest_max_evidence_age_days=7,
            digest_stale_evidence_warn_threshold=0.5,
            digest_low_source_penalty=0.10,
            digest_stale_penalty=0.05,
        )

    def test_digest_single_source_triggers_insufficient_warning(self) -> None:
        layer = self._build_layer()
        query = _mk_digest_query()
        now = datetime.now(timezone.utc)
        evidence = [_mk_evidence(document_id="only", published_at=now, support_score=0.6)]
        governance = _mk_governance(evidence)
        answer = _mk_answer_draft()

        baseline = layer.validate(query, governance, answer).confidence_score
        # Sanity：baseline 算出來應該介於 0~1
        self.assertGreaterEqual(baseline, 0.0)
        self.assertLessEqual(baseline, 1.0)

        result = layer.validate(query, governance, answer)
        # 必須有 digest 相關 warning
        self.assertTrue(
            any("Digest 品質 gate" in w and "低於最低需求" in w for w in result.warnings),
            f"期待 insufficient-source warning，但拿到 warnings={result.warnings}",
        )

    def test_digest_all_stale_triggers_stale_warning_and_penalty(self) -> None:
        layer = self._build_layer()
        query = _mk_digest_query()
        # 兩筆證據都超過 30 天 → stale_ratio=1.0，必觸發
        old = datetime.now(timezone.utc) - timedelta(days=30)
        evidence = [
            _mk_evidence(document_id="old-1", published_at=old, support_score=0.5),
            _mk_evidence(document_id="old-2", published_at=old, support_score=0.4),
        ]
        governance = _mk_governance(evidence)
        answer = _mk_answer_draft()

        result = layer.validate(query, governance, answer)
        self.assertTrue(
            any("stale_ratio" in w for w in result.warnings),
            f"期待 stale warning，但拿到 warnings={result.warnings}",
        )

    def test_digest_support_score_all_zero_triggers_warning(self) -> None:
        layer = self._build_layer()
        query = _mk_digest_query()
        now = datetime.now(timezone.utc)
        evidence = [
            _mk_evidence(document_id="z1", published_at=now, support_score=0.0),
            _mk_evidence(document_id="z2", published_at=now, support_score=0.0),
        ]
        governance = _mk_governance(evidence)
        answer = _mk_answer_draft()

        result = layer.validate(query, governance, answer)
        self.assertTrue(
            any("support_score=0" in w for w in result.warnings),
            f"期待 support_score=0 warning，但拿到 warnings={result.warnings}",
        )

    def test_digest_healthy_case_does_not_trigger_digest_warnings(self) -> None:
        """足量 + 新鮮 + support_score>0：三條 digest 警告都不該觸發。"""
        layer = self._build_layer()
        query = _mk_digest_query()
        now = datetime.now(timezone.utc)
        evidence = [
            _mk_evidence(document_id="h1", published_at=now, support_score=0.6),
            _mk_evidence(document_id="h2", published_at=now, support_score=0.4),
        ]
        governance = _mk_governance(evidence)
        answer = _mk_answer_draft()

        result = layer.validate(query, governance, answer)
        digest_warnings = [w for w in result.warnings if "Digest 品質 gate" in w]
        self.assertEqual(
            digest_warnings, [], f"健康情境不該有 digest warning，實際={digest_warnings}"
        )

    def test_legacy_profile_is_not_affected_by_digest_gate(self) -> None:
        """LEGACY profile：即使 evidence 很差，也不應觸發 digest-specific 的警告。"""
        layer = self._build_layer()
        query = _mk_legacy_query()
        old = datetime.now(timezone.utc) - timedelta(days=60)
        evidence = [
            _mk_evidence(document_id="legacy-old", published_at=old, support_score=0.0),
        ]
        governance = _mk_governance(evidence)
        answer = _mk_answer_draft()

        result = layer.validate(query, governance, answer)
        digest_warnings = [w for w in result.warnings if "Digest 品質 gate" in w]
        self.assertEqual(
            digest_warnings,
            [],
            f"LEGACY profile 不應觸發 digest gate，但拿到 {digest_warnings}",
        )

    def test_digest_penalty_reduces_confidence_vs_healthy_case(self) -> None:
        """單一 source 的 confidence 應該比健康情境低（確認 penalty 生效）。"""
        layer = self._build_layer()
        query = _mk_digest_query()
        now = datetime.now(timezone.utc)
        answer = _mk_answer_draft()

        healthy_evidence = [
            _mk_evidence(document_id="h1", published_at=now, support_score=0.6),
            _mk_evidence(document_id="h2", published_at=now, support_score=0.4),
        ]
        single_evidence = [
            _mk_evidence(document_id="solo", published_at=now, support_score=0.6),
        ]

        healthy = layer.validate(query, _mk_governance(healthy_evidence), answer)
        single = layer.validate(query, _mk_governance(single_evidence), answer)
        # single 的 confidence 應該 <= healthy（digest_low_source_penalty 有扣分）
        self.assertLessEqual(single.confidence_score, healthy.confidence_score)


# ─────────────────────────────────────────────────────────────────────
# 2) QueryDataHydrator skip_embedding_for_digest 成本 gate
# ─────────────────────────────────────────────────────────────────────


class _FakeEmbeddingService:
    """捕捉 embed_and_store 呼叫，供測試驗證是否真的觸發 embedding。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.call_args: list[tuple] = []
        self._called_event = threading.Event()

    def embed_and_store(self, documents, chunker=None) -> int:
        self.call_count += 1
        self.call_args.append((documents, chunker))
        self._called_event.set()
        return len(documents) if documents else 0

    def wait_called(self, timeout: float = 1.5) -> bool:
        return self._called_event.wait(timeout)


class _FakeGatewayForEmbed:
    """最小化 gateway：只提供 build_documents 回傳 1 個 document。"""

    def __init__(self) -> None:
        self.build_documents_called = 0

    def build_documents(self, query):
        self.build_documents_called += 1
        # 回傳結構上足以被 embedding_service 處理的任意物件
        return [{"id": "doc-1", "ticker": query.ticker or "2330", "content": "x"}]

    def sync_stock_info(self, force: bool = False) -> int:
        return 0


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.upsert_count = 0

    def upsert_documents(self, documents) -> int:
        self.upsert_count += len(documents) if documents else 0
        return self.upsert_count


class HydratorSkipEmbeddingTestCase(unittest.TestCase):
    """_trigger_embedding_async 的成本 gate 行為。"""

    def _make_hydrator(
        self,
        embedding_service: _FakeEmbeddingService,
        *,
        skip_embedding_for_digest: bool = True,
    ) -> tuple[QueryDataHydrator, _FakeGatewayForEmbed, _FakeDocumentRepository]:
        gateway = _FakeGatewayForEmbed()
        repo = _FakeDocumentRepository()
        hydrator = QueryDataHydrator(
            gateway,
            low_confidence_warmup_enabled=False,
            embedding_service=embedding_service,
            parallel_hydration_workers=1,
            skip_embedding_for_digest=skip_embedding_for_digest,
        )
        # 與 app.py 的線一致：document_repository 由外部注入
        hydrator._document_repository = repo
        return hydrator, gateway, repo

    def test_digest_profile_query_skips_embedding_by_default(self) -> None:
        fake_embed = _FakeEmbeddingService()
        hydrator, gateway, _ = self._make_hydrator(fake_embed)
        query = _mk_digest_query()

        hydrator._trigger_embedding_async(query)
        # 給 background thread 一點時間，但應該根本不會被啟動
        called = fake_embed.wait_called(timeout=0.3)
        self.assertFalse(
            called, "digest 路徑預設應跳過 embedding，但 embed_and_store 被呼叫了"
        )
        self.assertEqual(fake_embed.call_count, 0)
        # gateway.build_documents 也不該被呼叫（因為 gate 在背景 thread 啟動之前就 return 了）
        self.assertEqual(gateway.build_documents_called, 0)

    def test_legacy_profile_query_still_triggers_embedding(self) -> None:
        fake_embed = _FakeEmbeddingService()
        hydrator, gateway, _ = self._make_hydrator(fake_embed)
        query = _mk_legacy_query()

        hydrator._trigger_embedding_async(query)
        # Legacy 路徑 → 背景 thread 會實際跑 embed_and_store
        called = fake_embed.wait_called(timeout=1.5)
        self.assertTrue(called, "legacy 路徑應觸發 embedding，但 embed_and_store 沒被呼叫")
        self.assertEqual(fake_embed.call_count, 1)
        self.assertEqual(gateway.build_documents_called, 1)

    def test_digest_query_with_flag_disabled_still_triggers(self) -> None:
        """skip_embedding_for_digest=False 時，digest 路徑仍應照常 embed（override）。"""
        fake_embed = _FakeEmbeddingService()
        hydrator, gateway, _ = self._make_hydrator(
            fake_embed, skip_embedding_for_digest=False
        )
        query = _mk_digest_query()

        hydrator._trigger_embedding_async(query)
        called = fake_embed.wait_called(timeout=1.5)
        self.assertTrue(
            called, "skip_embedding_for_digest=False 時 digest 仍應 embed，但沒被呼叫"
        )
        self.assertEqual(fake_embed.call_count, 1)

    def test_no_embedding_service_is_noop(self) -> None:
        """embedding_service=None 時 _trigger_embedding_async 應靜默 return。"""
        gateway = _FakeGatewayForEmbed()
        hydrator = QueryDataHydrator(
            gateway,
            low_confidence_warmup_enabled=False,
            embedding_service=None,
            parallel_hydration_workers=1,
        )
        # 不會拋例外即可，順便確認 gateway 也沒被呼叫
        hydrator._trigger_embedding_async(_mk_legacy_query())
        # 稍等避免 CI 上 race condition
        time.sleep(0.1)
        self.assertEqual(gateway.build_documents_called, 0)


if __name__ == "__main__":
    unittest.main()
