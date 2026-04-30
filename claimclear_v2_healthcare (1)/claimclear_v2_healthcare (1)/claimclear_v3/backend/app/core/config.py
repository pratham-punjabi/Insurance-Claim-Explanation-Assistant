"""Central configuration — healthcare insurance assistant"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # ChromaDB — local persistent (default) or cloud
    chroma_host: str = ""           # empty = use local persisted DB
    chroma_port: int = 443
    chroma_ssl: bool = True
    chroma_api_key: str = ""
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "healthcare_policies"

    # Paths
    policies_dir: str = "data/policies"

    # App
    app_env: str = "development"
    cors_origins: str = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000"

    @property
    def policies_path(self) -> Path:
        return Path(self.policies_dir)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def use_cloud_chroma(self) -> bool:
        return bool(self.chroma_host)


settings = Settings()
