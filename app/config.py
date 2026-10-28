from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_api_key: SecretStr = Field(
        validation_alias=AliasChoices("LLM_API_KEY", "DASHSCOPE_API_KEY")
    )
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    product_db_path: Path = Path("data/products.db")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "products"
    order_service_base_url: str = "http://127.0.0.1:9000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
