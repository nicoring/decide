from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://127.0.0.1:11434/v1"
    model_name: str = "gpt-oss:20b"
    temperature: float = 0.3

    logfire_token: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
