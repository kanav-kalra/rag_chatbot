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

# Third-party imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local application imports
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This is the modern FastAPI approach (replaces deprecated @app.on_event).
    """
    # Startup
    logger.info("FastAPI application startup initiated.")
    
    # Initialize HR chatbot vector store (if HR chatbot is being used)
    try:
        from app.chatbot.hr_chatbot import initialize_hr_chatbot_vector_store
        initialize_hr_chatbot_vector_store()
    except ImportError:
        logger.debug("HR chatbot module not available, skipping vector store initialization.")
    except Exception as e:
        logger.warning(f"HR chatbot vector store initialization failed: {e}. Continuing without it.")
    
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
    streamlit_file = Path(__file__).parent / "streamlit_app.py"
    subprocess.run([
        "streamlit", "run", str(streamlit_file),
        "--server.port", str(streamlit_port),
        "--server.address", settings.HOST,
    ])


def run_fastapi():
    """Run FastAPI app with uvicorn"""
    uvicorn.run(
        "app.main:app",
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

