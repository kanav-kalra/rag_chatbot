"""
Application Configuration
"""
import os
from pathlib import Path
from typing import List

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


# Create settings instance
settings = Settings()

