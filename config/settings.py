"""
Centralized configuration management for BOQTenders.
All configurable parameters are defined here with sensible defaults.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


class LLMSettings(BaseSettings):
    """LLM-related configuration."""
    
    # API Keys
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"),description="Google API key for Gemini")
    
    # Model Configuration
    model_name: str = Field(default="gemini-2.5-flash-lite",description="LLM model name to use")
    temperature: float = Field(default=0.0,ge=0.0,le=2.0,description="LLM temperature (0.0 = deterministic)")
    max_output_tokens: int = Field(default=8192,description="Maximum tokens in LLM response")


class EmbeddingSettings(BaseSettings):
    """Embedding and vector store configuration."""
    
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2",description="HuggingFace embedding model name")
    
    # Text Splitting
    chunk_size: int = Field(default=1000,ge=100,le=10000,description="Size of text chunks for splitting")
    chunk_overlap: int = Field(default=500,ge=0,description="Overlap between consecutive chunks")


class PDFExtractionSettings(BaseSettings):
    """PDF extraction API configuration."""
    
    hf_api_token: str = Field(default_factory=lambda: os.getenv("HF_API_TOKEN"),description="HuggingFace API token")
    extraction_api_url: str = Field(default="https://point9-extract-text-and-table.hf.space/api/text",description="URL for PDF text extraction API")
    
    start_page: int = Field(default=1,ge=1,description="Default start page for extraction")
    end_page: int = Field(default=100,ge=1,description="Default end page for extraction")
    request_timeout: int = Field(default=120,description="API request timeout in seconds")


class BOQExtractionSettings(BaseSettings):
    """BOQ extraction specific configuration."""
    
    batch_size: int = Field(default=25,ge=1,le=100,description="Number of chunks per batch for BOQ extraction")
    max_prompt_length: int = Field(default=30000,description="Maximum characters in extraction prompt")
    page_search_length: int = Field(default=30,description="Characters to use for page detection search")
    source_max_length: int = Field(default=50,description="Maximum length for source column")


class ConsistencySettings(BaseSettings):
    """Consistency check configuration."""
    
    default_runs: int = Field(default=4,ge=2,le=10,description="Default number of runs for consistency check")
    low_consistency_threshold: float = Field(default=80.0,ge=0.0,le=100.0,description="Threshold below which consistency is considered low")


class APISettings(BaseSettings):
    """FastAPI server configuration."""
    
    title: str = Field(default="BOQ Chatbot API",description="API title")
    description: str = Field(default="API for extracting and querying BOQ from tender PDFs using RAG",description="API description")
    version: str = Field(default="1.0.0",description="API version")
    host: str = Field(default="0.0.0.0",description="Server host")
    port: int = Field(default=8000,description="Server port")
    debug: bool = Field(default=False,description="Enable debug mode")
    docs_enabled: bool = Field(default=True,description="Enable API documentation endpoints")
    cors_origins: list = Field(default=["*"],description="Allowed CORS origins")


class StreamlitSettings(BaseSettings):
    """Streamlit UI configuration."""
    
    page_title: str = Field(default="BOQ Agent",description="Page title")
    page_icon: str = Field(default="📄",description="Page icon")
    layout: str = Field(default="wide",description="Page layout (wide/centered)")


class Settings(BaseSettings):
    """
    Main settings class that aggregates all configuration sections.
    Access via: settings.llm, settings.embedding, settings.pdf, etc.
    """
    
    # Global settings
    log_level: str = Field(default="INFO",description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pdf: PDFExtractionSettings = Field(default_factory=PDFExtractionSettings)
    boq: BOQExtractionSettings = Field(default_factory=BOQExtractionSettings)
    consistency: ConsistencySettings = Field(default_factory=ConsistencySettings)
    api: APISettings = Field(default_factory=APISettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars not defined in model


# Global settings instance
settings = Settings()
