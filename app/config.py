"""Application settings."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    fact_version: str = "speaker-catalogue-v4"

    expected_answers: int = 50
    expected_facts: int = 132
    expected_gold: int = 376

    @property
    def answers_path(self) -> Path:
        return self.data_dir / "answers.csv"

    @property
    def brand_facts_path(self) -> Path:
        return self.data_dir / "brand_facts.csv"

    @property
    def gold_path(self) -> Path:
        return self.data_dir / "development_gold.csv"


def get_settings() -> Settings:
    return Settings()
