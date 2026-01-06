"""
BOQTenders API Server

FastAPI application entry point for BOQ extraction and chat services.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.settings import settings

# Create FastAPI app
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url="/docs" if settings.api.docs_enabled else None,
    redoc_url="/redoc" if settings.api.docs_enabled else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Import and include routes
from api.routes import router
app.include_router(router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for HuggingFace Spaces."""
    return {"status": "healthy"}

# Export app for uvicorn
__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
    )
