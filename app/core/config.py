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
    # Supported models: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "claude-3-opus", "claude-3-sonnet", "gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    CHAT_MODEL_TEMPERATURE: float = float(os.getenv("CHAT_MODEL_TEMPERATURE", "0.7"))
    CHAT_MODEL_MAX_TOKENS: int = int(os.getenv("CHAT_MODEL_MAX_TOKENS", "2000"))
    
    # Anthropic Configuration (optional)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Google Gemini Configuration (optional)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
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
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Memory Management Configuration
    # Memory strategies: "none", "trim", "summarize", "trim_and_summarize"
    DEFAULT_MEMORY_STRATEGY: str = os.getenv("DEFAULT_MEMORY_STRATEGY", "none")
    MEMORY_TRIM_KEEP_MESSAGES: int = int(os.getenv("MEMORY_TRIM_KEEP_MESSAGES", "10"))  # Keep last N messages
    MEMORY_SUMMARIZE_THRESHOLD: int = int(os.getenv("MEMORY_SUMMARIZE_THRESHOLD", "20"))  # Summarize when messages exceed this
    MEMORY_SUMMARIZE_MODEL: str = os.getenv("MEMORY_SUMMARIZE_MODEL", "gpt-3.5-turbo-16k")  # Model for summarization (should have high context window)
    
    # Note: HR chatbot configuration (model, agent pool, vector store) is now in hr_chatbot_config.yaml
    # The vector store manager will read from YAML config first, then fall back to defaults
    
    # LangSmith Configuration (for observability and tracing)
    # Note: Both LANGSMITH_API_KEY and LANGCHAIN_API_KEY are supported
    # LANGSMITH_API_KEY is the preferred name per LangSmith Studio documentation
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    # Support both LANGSMITH_API_KEY (preferred) and LANGCHAIN_API_KEY (legacy)
    LANGCHAIN_API_KEY: str = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "rag-chatbot")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # Workspace ID is only required for org-scoped API keys
    # If your API key is org-scoped, set this to your workspace ID
    # If not org-scoped, leave it empty
    LANGSMITH_WORKSPACE_ID: str = os.getenv("LANGSMITH_WORKSPACE_ID", "")


# Create settings instance
settings = Settings()

