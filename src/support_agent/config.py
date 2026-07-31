"""Application settings, loaded from the environment / `.env`.

Field names mirror `.env.example` one for one. Everything tunable lives here so
no module reaches for `os.environ` directly and no limit is hardcoded in the
agent loop.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider -------------------------------------------------------
    openai_api_key: str
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Database -----------------------------------------------------------
    database_url: str

    # --- Session lifecycle --------------------------------------------------
    session_idle_ttl_minutes: int = 30
    session_absolute_ttl_hours: int = 24
    session_sweep_interval_seconds: int = 300

    # --- Agent limits -------------------------------------------------------
    agent_max_iterations: int = 6
    escalation_tool_failure_threshold: int = 2
    escalation_min_retrieval_score: float = 0.35

    # --- Observability ------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton. Cached so `.env` is read once."""
    return Settings()
