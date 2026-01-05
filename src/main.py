"""
Main FastAPI Application
"""
# Standard library imports
import argparse
import logging
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file FIRST, before any other imports
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from src.api.v1.router import api_router
from src.shared.config.settings import settings
from src.shared.config.logging import logger
from src.api.middleware.rate_limiter import RateLimitMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This is the modern FastAPI approach (replaces deprecated @app.on_event).
    """
    # Startup
    logger.info("FastAPI application startup initiated.")
    
    # Initialize LangSmith tracing (if enabled)
    try:
        from src.shared.config.langsmith import initialize_langsmith
        if initialize_langsmith():
            logger.info("LangSmith tracing enabled")
        else:
            logger.info("LangSmith tracing is disabled (set LANGCHAIN_TRACING_V2=true to enable)")
    except Exception as e:
        logger.warning(f"LangSmith initialization failed: {e}. Continuing without tracing.")
    
    # Initialize checkpointer manager (Redis connection)
    try:
        from src.infrastructure.storage.checkpointing.manager import get_checkpointer_manager
        manager = get_checkpointer_manager()
        logger.info(f"Checkpointer initialized: {'Redis' if manager.is_redis else 'In-Memory'}")
    except Exception as e:
        logger.warning(f"Checkpointer initialization failed: {e}. Continuing with fallback.")
    
    # Note: Chatbot vector stores are now loaded on-demand from ChromaDB
    # Run 'python jobs/create_vectorstore.py --chatbot-type <type>' to create a vector store
    
    yield
    
    # Shutdown (if needed)
    # Clean up resources here
    logger.info("FastAPI application shutdown.")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.PROJECT_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,  # Modern lifespan event handler
    )

    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware (only in production)
    if not settings.DEBUG:
        application.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            requests_per_hour=1000
        )
        logger.info("Rate limiting middleware enabled")

    # Include API routers
    application.include_router(api_router, prefix=settings.API_PREFIX)

    return application


# Create the app instance
app = create_application()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to FastAPI",
        "version": settings.PROJECT_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


def run_streamlit():
    """Run Streamlit app in a subprocess"""
    streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
    streamlit_file = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([
        "streamlit", "run", str(streamlit_file),
        "--server.port", str(streamlit_port),
        "--server.address", settings.HOST,
    ])


def run_fastapi():
    """Run FastAPI app with uvicorn"""
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI and/or Streamlit applications")
    parser.add_argument(
        "--app",
        choices=["fastapi", "streamlit", "both"],
        default="both",
        help="Which application to run (default: both)"
    )
    args = parser.parse_args()
    
    if args.app == "fastapi":
        run_fastapi()
    elif args.app == "streamlit":
        run_streamlit()
    else:  # both
        # Run FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Run Streamlit in the main thread (blocking)
        logger.info(f"Starting FastAPI on http://{settings.HOST}:{settings.PORT}")
        logger.info(f"Starting Streamlit on http://{settings.HOST}:{os.getenv('STREAMLIT_PORT', '8501')}")
        run_streamlit()

