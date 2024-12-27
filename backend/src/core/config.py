from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 8000

    RUN_LOCAL_LLM: bool = True
    LOCAL_LLM_HOST: str = "localhost"
    LOCAL_LLM_PORT: int = 11434

    LOCAL_MODEL: str = "llama3.1:8b"
    API_MODEL: str

    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    LANGCHAIN_DEBUG: bool = True

    DATA_DIR: str = "./data/"

    LOG_LEVEL: str = "INFO"
    # LOG_FILE: Optional[str] = None  # "logs/app.log"

    class Config:
        env_file = ".env"  # debug-only
        case_sensitive = True


config = Settings()
