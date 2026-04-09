from pathlib import Path

from llm_stock_system.core.interfaces import LLMClient
from llm_stock_system.core.models import AnswerDraft, GovernanceReport, StructuredQuery


class GenerationLayer:
    def __init__(self, llm_client: LLMClient, prompt_path: Path) -> None:
        self._llm_client = llm_client
        self._prompt_path = prompt_path

    def generate(self, query: StructuredQuery, governance_report: GovernanceReport) -> AnswerDraft:
        system_prompt = self._load_prompt()
        return self._llm_client.synthesize(query, governance_report, system_prompt)

    def _load_prompt(self) -> str:
        return self._prompt_path.read_text(encoding="utf-8")
