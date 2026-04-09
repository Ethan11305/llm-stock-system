from enum import Enum


class Topic(str, Enum):
    NEWS = "news"
    EARNINGS = "earnings"
    ANNOUNCEMENT = "announcement"
    COMPOSITE = "composite"


class SourceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StanceBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ConfidenceLight(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    MOSTLY_CONSISTENT = "mostly_consistent"
    CONFLICTING = "conflicting"


class FreshnessStatus(str, Enum):
    RECENT = "recent"
    STALE = "stale"
    OUTDATED = "outdated"
