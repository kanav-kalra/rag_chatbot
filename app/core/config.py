"""
Application Configuration
"""
import os
from pathlib import Path
from typing import List, Optional

# Load environment variables from .env file if it exists
from dotenv import load_dotenv

# Get the project root directory (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"

# Load .env file if it exists
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables"""

    # Project Information
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "RAG chatbot")
    PROJECT_DESCRIPTION: str = os.getenv("PROJECT_DESCRIPTION", "A RAG chatbot application")
    PROJECT_VERSION: str = os.getenv("PROJECT_VERSION", "1.0.0")

    # API Configuration
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS Configuration
    CORS_ORIGINS: List[str] = (
        os.getenv("CORS_ORIGINS", "*").split(",")
        if os.getenv("CORS_ORIGINS") != "*"
        else ["*"]
    )

    # Database Configuration (optional)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LLM Model Configuration
    # Supported models: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "gpt-4o"
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    CHAT_MODEL_TEMPERATURE: float = float(os.getenv("CHAT_MODEL_TEMPERATURE", "0.7"))
    CHAT_MODEL_MAX_TOKENS: int = int(os.getenv("CHAT_MODEL_MAX_TOKENS", "2000"))
    
    # Anthropic Configuration (optional)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Ollama Configuration (optional)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Session Management Configuration
    MAX_CONCURRENT_SESSIONS: Optional[int] = (
        int(os.getenv("MAX_CONCURRENT_SESSIONS"))
        if os.getenv("MAX_CONCURRENT_SESSIONS")
        else None
    )
    SESSION_TIMEOUT_HOURS: int = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
    AGENT_POOL_SIZE: int = int(os.getenv("AGENT_POOL_SIZE", "1"))  # Default: 1 shared agent


# Create settings instance
settings = Settings()

