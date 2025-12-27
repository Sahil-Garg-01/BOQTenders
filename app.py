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

from loguru import logger
from config.settings import settings
from api.routes import app

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Export app for uvicorn
__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
