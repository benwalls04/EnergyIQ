from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    duckdb_path: str = "/data/app.duckdb"
    cors_origins: str = "http://localhost:5173"

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
