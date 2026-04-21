"""Digest 產品線主要路徑。

本 package 為 `single_stock_digest` 專屬實作。依照
`digest_refusal_boundary_v1_1.md`，digest 路徑不得依賴下列 legacy 能力：

- `core.models.infer_intent_from_question_type`
- `core.models.QUESTION_TYPE_FALLBACK_TOPIC_TAGS`
- 任意 `question_type -> intent` 反推

reuse 的共用能力只限於：enums、Evidence、GovernanceReport 等純資料結構。
"""
