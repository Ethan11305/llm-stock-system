import logging
from datetime import datetime, timedelta, timezone

from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, QueryProfile, StanceBias
from llm_stock_system.core.models import AnswerDraft, GovernanceReport, StructuredQuery, ValidationResult
from llm_stock_system.core.validation_profiles import ConditionKind, ValidationProfile, ValidationRule, get_profile

logger = logging.getLogger(__name__)


class ValidationLayer:
    _PRELIMINARY_PREFIXES = ("初步判讀：", "preliminary")
    _INSUFFICIENT_DATA_TOKENS = ("資料不足", "無法確認")

    def __init__(
        self,
        min_green_confidence: float,
        min_yellow_confidence: float,
        *,
        digest_min_sources: int = 2,
        digest_max_evidence_age_days: int = 7,
        digest_stale_evidence_warn_threshold: float = 0.5,
        digest_low_source_penalty: float = 0.10,
        digest_stale_penalty: float = 0.05,
    ) -> None:
        self._min_green_confidence = min_green_confidence
        self._min_yellow_confidence = min_yellow_confidence
        # P2 digest-specific quality gate 門檻
        self._digest_min_sources = digest_min_sources
        self._digest_max_evidence_age_days = digest_max_evidence_age_days
        self._digest_stale_warn_threshold = digest_stale_evidence_warn_threshold
        self._digest_low_source_penalty = digest_low_source_penalty
        self._digest_stale_penalty = digest_stale_penalty

    def validate(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        facet_miss_list: list[str] | None = None,
        preferred_miss_list: list[str] | None = None,
        hydration_warnings: list[str] | None = None,
    ) -> ValidationResult:
        confidence_score = self._calculate_base_confidence(governance_report, answer_draft)
        # hydration_warnings 優先放入（例如 FinMind rate limit），確保使用者最先看到
        warnings: list[str] = list(hydration_warnings or [])

        confidence_score = self._apply_general_checks(
            query,
            governance_report,
            answer_draft,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_required_facet_cap(
            query,
            facet_miss_list,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_preferred_facet_penalty(
            query,
            preferred_miss_list,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_intent_profile_rules(
            query,
            governance_report,
            answer_draft,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_match_type_cap(
            query,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_digest_profile_gate(
            query,
            governance_report,
            confidence_score,
            warnings,
        )
        confidence_score = round(max(confidence_score, 0.0), 2)

        if confidence_score >= self._min_green_confidence:
            light = ConfidenceLight.GREEN
            status = "pass"
        elif confidence_score >= self._min_yellow_confidence:
            light = ConfidenceLight.YELLOW
            status = "review"
        else:
            light = ConfidenceLight.RED
            status = "blocked"

        return ValidationResult(
            confidence_score=confidence_score,
            confidence_light=light,
            validation_status=status,
            warnings=warnings,
            facet_miss_list=list(facet_miss_list or []),
        )

    def _calculate_base_confidence(
        self,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
    ) -> float:
        n = len(governance_report.evidence)
        # Dynamic denominator: treat 2 pieces of evidence as the baseline
        # for a well-supported answer.  A fixed denominator of 4 unfairly
        # penalises small-cap stocks where the maximum retrievable evidence
        # is naturally 2–3 items.  With max(n, 2) the score reaches 1.0 as
        # soon as we have ≥ 2 sources; a single document still scores 0.5.
        evidence_score = min(n / max(n, 2), 1.0) * 0.30
        trust_score = governance_report.high_trust_ratio * 0.25
        freshness_score = self._freshness_weight(governance_report.freshness) * 0.20
        consistency_score = self._consistency_weight(governance_report.consistency) * 0.15
        citation_score = min(len(answer_draft.sources) / max(n, 1), 1.0) * 0.10
        return round(
            evidence_score + trust_score + freshness_score + consistency_score + citation_score,
            2,
        )

    def _apply_general_checks(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        summary = answer_draft.summary.strip()

        if query.stance_bias != StanceBias.NEUTRAL and len(answer_draft.risks) < 3:
            warnings.append("Biased question should include at least three risks.")
            confidence_score = max(confidence_score - 0.10, 0.0)
        if not governance_report.evidence:
            warnings.append("No supporting evidence retrieved.")
        if self._is_preliminary_summary(summary):
            warnings.append("Preliminary LLM answer returned without grounded local evidence.")
            confidence_score = min(confidence_score, 0.35)
        if governance_report.consistency == ConsistencyStatus.CONFLICTING:
            warnings.append("Evidence consistency is low.")
        if any(token in summary for token in self._INSUFFICIENT_DATA_TOKENS):
            warnings.append("Answer indicates insufficient data.")
            confidence_score = min(confidence_score, 0.25)

        # P2 Integration：從 PolicyRegistry 取得 min_evidence_count。
        # 若 evidence 數量低於該策略的最低要求，降至 YELLOW 上限（0.75）並加 warning。
        confidence_score = self._apply_policy_evidence_floor(
            query, governance_report, confidence_score, warnings
        )

        return confidence_score

    def _apply_policy_evidence_floor(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """檢查本次查詢的 policy 是否要求最低 evidence 數量。

        使用 intent + controlled_tags 路由（不依賴 question_type），
        與 Wave 4 的 routing 主軸一致。

        若 evidence 數量不足 policy.min_evidence_count，
        將信心分上限壓到 YELLOW（0.75），並加 warning 說明缺少幾篇。
        """
        try:
            from llm_stock_system.core.query_policy import get_policy_registry
            policy = get_policy_registry().resolve_by_tags(
                query.intent, query.controlled_tags or []
            )
            min_count = policy.min_evidence_count
            actual_count = len(governance_report.evidence)
            if min_count > 1 and actual_count < min_count:
                cap = 0.75  # YELLOW 上限
                if confidence_score > cap:
                    confidence_score = cap
                # 以 intent 名稱標記查詢類型（warning 只作說明，不影響路由）
                warnings.append(
                    f"此查詢類型（{query.intent.value}）需要至少 {min_count} 篇證據，"
                    f"目前只有 {actual_count} 篇。信心分上限 {cap}。"
                )
        except Exception:
            # Registry 不可用時靜默忽略，不影響現有邏輯
            pass
        return confidence_score

    def _apply_required_facet_cap(
        self,
        query: StructuredQuery,
        facet_miss_list: list[str] | None,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        required_facets = {facet.value for facet in query.required_facets}
        required_misses = sorted(set(facet_miss_list or []) & required_facets)

        if not required_facets or not required_misses:
            return confidence_score

        if len(required_misses) == len(required_facets):
            warnings.append(f"All required facets failed to sync: {required_misses}")
            return min(confidence_score, 0.25)

        warnings.append(f"Required facet sync failed (partial): {required_misses}")
        return min(confidence_score, 0.50)

    def _apply_preferred_facet_penalty(
        self,
        query: StructuredQuery,
        preferred_miss_list: list[str] | None,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        preferred_facets = {facet.value for facet in query.preferred_facets}
        preferred_misses = sorted(set(preferred_miss_list or []) & preferred_facets)

        if not preferred_misses:
            return confidence_score

        penalty = min(len(preferred_misses) * 0.10, 0.30)
        warnings.append(f"Preferred facets not synced ({len(preferred_misses)}): {preferred_misses}")
        return max(confidence_score - penalty, 0.0)

    def _evaluate_profile(
        self,
        profile: ValidationProfile,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """Evaluate an intent profile with optional topic-tag rule sets."""
        source_names = self._source_names(governance_report)
        combined_text = self._combined_text(governance_report)
        normalized_text = combined_text.lower()
        query_tags = set(query.topic_tags)

        for rule in profile.base_rules:
            confidence_score = self._apply_rule(
                rule,
                query,
                governance_report,
                answer_draft,
                source_names,
                normalized_text,
                confidence_score,
                warnings,
            )

        for tag_rule_set in profile.tag_rules:
            if not tag_rule_set.required_tags.issubset(query_tags):
                continue

            for rule in tag_rule_set.rules:
                confidence_score = self._apply_rule(
                    rule,
                    query,
                    governance_report,
                    answer_draft,
                    source_names,
                    normalized_text,
                    confidence_score,
                    warnings,
                )

            if tag_rule_set.custom_validator is not None:
                confidence_score = tag_rule_set.custom_validator(
                    query,
                    governance_report,
                    answer_draft,
                    confidence_score,
                    warnings,
                )
            break

        if profile.custom_validator is not None:
            confidence_score = profile.custom_validator(
                query,
                governance_report,
                answer_draft,
                confidence_score,
                warnings,
            )

        return confidence_score

    def _apply_rule(
        self,
        rule: ValidationRule,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        source_names: set[str],
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if rule.condition == ConditionKind.DUAL_SIGNAL_MISSING:
            return self._apply_dual_signal_rule(rule, source_names, confidence_score, warnings)

        if not self._is_profile_rule_triggered(
            rule,
            query,
            governance_report,
            answer_draft,
            source_names,
            normalized_text,
        ):
            return confidence_score

        if rule.warning:
            warnings.append(rule.warning)
        if rule.cap is not None:
            confidence_score = min(confidence_score, rule.cap)
        if rule.penalty is not None:
            confidence_score = max(confidence_score - rule.penalty, 0.0)
        return confidence_score

    def _is_profile_rule_triggered(
        self,
        rule: ValidationRule,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        source_names: set[str],
        normalized_text: str,
    ) -> bool:
        if rule.condition == ConditionKind.SOURCE_FRAGMENT_MISSING:
            fragments = tuple(rule.params.get("fragments", ()))
            return not self._has_source_fragment(source_names, *fragments)

        if rule.condition == ConditionKind.CONTENT_KEYWORD_MISSING:
            keywords = [str(keyword).lower() for keyword in rule.params.get("keywords", ())]
            match_mode = str(rule.params.get("match_mode", "any")).lower()
            if match_mode == "all":
                return any(keyword not in normalized_text for keyword in keywords)
            return not any(keyword in normalized_text for keyword in keywords)

        if rule.condition == ConditionKind.EVIDENCE_COUNT_BELOW:
            threshold = int(rule.params.get("threshold", 0))
            return len(governance_report.evidence) < threshold

        if rule.condition == ConditionKind.COMPARISON_COMPANY_MISSING:
            if not query.comparison_ticker:
                return False
            primary_label = (query.company_name or query.ticker or "").lower()
            comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
            if not (primary_label and comparison_label):
                return False
            return primary_label not in normalized_text or comparison_label not in normalized_text

        if rule.condition == ConditionKind.ANSWER_CONTAINS_TOKEN:
            summary = answer_draft.summary.lower()
            tokens = [str(token).lower() for token in rule.params.get("tokens", ())]
            return any(token in summary for token in tokens)

        raise ValueError(f"Unsupported condition kind: {rule.condition}")

    def _apply_dual_signal_rule(
        self,
        rule: ValidationRule,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        signal_a_fragments = tuple(rule.params.get("signal_a_fragments", ()))
        signal_b_fragments = tuple(rule.params.get("signal_b_fragments", ()))
        has_signal_a = self._has_source_fragment(source_names, *signal_a_fragments)
        has_signal_b = self._has_source_fragment(source_names, *signal_b_fragments)

        if has_signal_a and has_signal_b:
            return confidence_score

        if not has_signal_a and not has_signal_b:
            warning = str(rule.params.get("both_missing_warning", "")).strip()
            cap = rule.params.get("both_missing_cap")
        else:
            warning = str(rule.params.get("one_missing_warning", "")).strip()
            cap = rule.params.get("one_missing_cap")

        if warning:
            warnings.append(warning)
        if cap is not None:
            confidence_score = min(confidence_score, float(cap))
        return confidence_score

    def _apply_intent_profile_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """Apply intent-keyed validation profile rules.

        Wave 4 Stage 5 rename: previously `_apply_question_type_rules`. The
        method has always keyed on ``query.intent`` via :func:`get_profile`,
        so the rename only clarifies intent.
        """
        profile = get_profile(query.intent)
        if profile is None:
            return confidence_score
        return self._evaluate_profile(
            profile,
            query,
            governance_report,
            answer_draft,
            confidence_score,
            warnings,
        )

    def _apply_match_type_cap(
        self,
        query: StructuredQuery,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """規劃項目 B：根據 policy_match_type 調整信心分上限。

        routing 品質直接影響回答的可靠度：
          - "fallback" → policy 完全預設，路由沒有語意依據，信心分上限 0.72（YELLOW）
          - "generic"  → intent 命中但 tag 未匹配，稍微保守，上限 0.78（YELLOW 上緣）
          - "partial"  → 部分 tag 命中，不調整
          - "exact"    → 全部 tag 命中，不調整

        使用 controlled_tags 路由（不混 free_keywords），與規劃項目 A 的改動一致。

        注意：若 controlled_tags 為空，代表本次查詢沒有可路由的 enum tag，
        無法判斷路由品質，故直接跳過不調整。
        """
        if not query.controlled_tags:
            return confidence_score
        try:
            from llm_stock_system.core.query_policy import get_policy_registry
            policy = get_policy_registry().resolve_by_tags(
                query.intent, query.controlled_tags or []
            )
            match_type = policy.match_type
            if match_type == "fallback":
                cap = 0.72
                if confidence_score > cap:
                    confidence_score = cap
                    warnings.append(
                        "路由退回全域預設（policy fallback），信心分上限壓至 0.72。"
                    )
            elif match_type == "generic":
                cap = 0.78
                if confidence_score > cap:
                    confidence_score = cap
                    warnings.append(
                        "查詢未命中任何具體 policy（generic routing），信心分上限壓至 0.78。"
                    )
            # "partial" 和 "exact" 不調整
        except Exception:
            pass  # Registry 不可用時靜默忽略
        return confidence_score

    def _apply_digest_profile_gate(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """P2 Digest 產品線品質 gate。

        只對 ``query_profile == SINGLE_STOCK_DIGEST`` 的查詢生效，legacy 路徑完全不碰。

        規則（都是 warning-first，非 hard-fail）：
          1. evidence 數量 < digest_min_sources：加 warning 並扣 digest_low_source_penalty
          2. evidence 中已過時（published_at 早於 N 天前）的比例 ≥ threshold：
             加 warning 並扣 digest_stale_penalty
          3. 所有 evidence support_score 皆為 0：加 warning（通常代表沒有真正的對證據加權）

        設計理念：digest 產品對「新聞時效 + 交叉驗證」極度敏感，但使用者仍應能看到結果；
        因此用 warnings + 輕度降 confidence 引導前端 UI 呈現「需審閱」樣式，而不是直接擋掉。
        """
        if query.query_profile != QueryProfile.SINGLE_STOCK_DIGEST:
            return confidence_score

        evidence = governance_report.evidence
        n = len(evidence)

        # 1) 最低 source 要求
        if n < self._digest_min_sources:
            warnings.append(
                f"Digest 品質 gate：僅 {n} 筆 evidence，低於最低需求 {self._digest_min_sources}"
                "，前端應標示『資料單一，交叉驗證不足』。"
            )
            confidence_score = max(confidence_score - self._digest_low_source_penalty, 0.0)

        # 2) evidence 時效性
        if n > 0:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=self._digest_max_evidence_age_days)
            stale_count = 0
            for item in evidence:
                published_at = item.published_at
                # 容錯：published_at 無 tzinfo 時視為 UTC
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if published_at < cutoff:
                    stale_count += 1
            stale_ratio = stale_count / n
            if stale_ratio >= self._digest_stale_warn_threshold:
                warnings.append(
                    f"Digest 品質 gate：{stale_count}/{n} 筆證據超過 "
                    f"{self._digest_max_evidence_age_days} 天（stale_ratio="
                    f"{stale_ratio:.0%}），與 digest 的 7 天時間窗不符。"
                )
                confidence_score = max(confidence_score - self._digest_stale_penalty, 0.0)

            # 3) support_score 全 0 檢查
            if all(item.support_score == 0 for item in evidence):
                warnings.append(
                    "Digest 品質 gate：全部 evidence 的 support_score=0，governance 沒能對證據加權。"
                )

        return confidence_score

    # --- Helper methods ---

    def _source_names(self, governance_report: GovernanceReport) -> set[str]:
        return {item.source_name.lower() for item in governance_report.evidence}

    def _combined_text(self, governance_report: GovernanceReport) -> str:
        return " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)

    def _has_source_fragment(self, source_names: set[str], *fragments: str) -> bool:
        return any(fragment in source_name for fragment in fragments for source_name in source_names)

    def _is_preliminary_summary(self, summary: str) -> bool:
        return any(summary.startswith(prefix) for prefix in self._PRELIMINARY_PREFIXES)

    def _freshness_weight(self, status: FreshnessStatus) -> float:
        return {
            FreshnessStatus.RECENT: 1.0,
            FreshnessStatus.STALE: 0.6,
            FreshnessStatus.OUTDATED: 0.2,
        }.get(status, 0.0)

    def _consistency_weight(self, status: ConsistencyStatus) -> float:
        return {
            ConsistencyStatus.CONSISTENT: 1.0,
            ConsistencyStatus.MOSTLY_CONSISTENT: 0.6,
            ConsistencyStatus.CONFLICTING: 0.2,
        }.get(status, 0.0)
