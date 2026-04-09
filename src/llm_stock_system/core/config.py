from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLM Stock Advisory System"
    api_prefix: str = "/api"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock"
    use_postgres_market_data: bool = True
    fallback_to_sample_data: bool = True
    finmind_base_url: str = "https://api.finmindtrade.com/api/v4"
    finmind_api_token: str = ""
    finmind_sync_on_query: bool = True
    stock_info_refresh_hours: int = 24
    news_pipeline_enabled: bool = True
    google_news_rss_enabled: bool = True
    google_news_rss_base_url: str = "https://news.google.com/rss/search"
    twse_company_financial_url: str = "https://www.twse.com.tw/rwd/zh/IIH/company/financial"
    twse_monthly_revenue_url: str = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
    openai_base_url: str = ""
    openai_api_key: str = ""
    preliminary_llm_answers_enabled: bool = True
    low_confidence_warmup_enabled: bool = True
    low_confidence_warmup_threshold: float = 0.80
    low_confidence_warmup_cooldown_hours: int = 12
    model_name: str = "gpt-4.1-mini"
    max_retrieval_docs: int = 8
    min_green_confidence: float = 0.80
    min_yellow_confidence: float = 0.55

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def prompt_path(self) -> Path:
        return self.project_root / "src" / "llm_stock_system" / "prompts" / "system_prompt.md"


@lru_cache
def get_settings() -> Settings:
    return Settings()
