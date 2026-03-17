"""
BOQTenders API Server

FastAPI application entry point for BOQ extraction and chat services.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from config.settings import settings
import os

# ============================================================================
# OpenTelemetry Setup (Traces → Tempo | Logs → Loki | Metrics → Prometheus)
# ============================================================================
otel_enabled = False
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    
    # Instrumentation
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
    # Note: Logging instrumentation is handled by opentelemetry-distro

    # Resource definition
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "boqtenders"),
        "service.version": "1.0.0",
        "telemetry.sdk.language": "python",
    })
    
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # ========== TRACES ==========
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    sampler_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    
    trace_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(sampler_ratio),
    )
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(trace_provider)
    tracer = trace.get_tracer(__name__)
    
    # ========== METRICS ==========
    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
    metric_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(metric_exporter)],
    )
    metrics.set_meter_provider(metric_provider)
    meter = metrics.get_meter(__name__)
    
    # ========== LOGS ==========
    # Logging instrumentation is handled automatically by opentelemetry-distro
    # when using standard Python logging or loguru
    
    # ========== LIBRARY INSTRUMENTATION ==========
    RequestsInstrumentor().instrument()
    PymongoInstrumentor().instrument()
    
    otel_enabled = True
    logger.info("✓ OpenTelemetry initialized | Traces→Tempo | Logs→Loki | Metrics→Prometheus")

except Exception as e:
    logger.warning(f"⚠ OpenTelemetry warning: {e}")
    otel_enabled = False

# Create FastAPI app
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url="/docs" if settings.api.docs_enabled else None,
    redoc_url="/redoc" if settings.api.docs_enabled else None,
)

# Instrument FastAPI (only if OTEL loaded)
if otel_enabled:
    FastAPIInstrumentor.instrument_app(app)

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

# Root endpoint for Spaces health check
@app.get("/")
async def root():
    """Root endpoint for Hugging Face Spaces health check."""
    return {"message": "BOQ Tenders Agent API", "status": "running", "docs": "/docs"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for HuggingFace Spaces."""
    return {"status": "healthy"}

# Grafana iframe endpoint
@app.get("/grafana-iframe", response_class=HTMLResponse)
async def get_grafana_iframe(dashboard_uid: str, name: str = "BOQ Monitoring", height: int = 600, width: str = "100%"):
    """Get Grafana dashboard iframe HTML for frontend integration.
    
    Returns HTML iframe tag with proper Content-Type: text/html.
    Frontend can embed directly via innerHTML or fetch + direct insertion.
    
    Args:
        dashboard_uid: Grafana dashboard UID (required)
        name: Dashboard display name (default: "BOQ Monitoring")
        height: Iframe height in pixels (default: 600)
        width: Iframe width (default: "100%")
    
    Returns:
        HTML: Iframe tag that embeds the Grafana dashboard
        
    Example:
        GET /grafana-iframe?dashboard_uid=abc123&name=BOQ%20Dashboard
        Returns: <iframe src="http://grafana:3000/d/abc123/boq-dashboard..." ...>
    """
    if not dashboard_uid:
        return "<div style='padding: 20px; color: red;'><strong>Error:</strong> dashboard_uid parameter is required</div>"
    
    try:
        from monitoring_api import GrafanaEmbedder
        embedder = GrafanaEmbedder()
        html = embedder.get_dashboard_iframe(dashboard_uid, name, height=height, width=width)
        
        logger.info(f"✓ Generated Grafana iframe for dashboard: {dashboard_uid}")
        return html  # Returns HTML with Content-Type: text/html
        
    except Exception as e:
        logger.error(f'❌ Failed to generate iframe for dashboard {dashboard_uid}: {e}', exc_info=True)
        return f"<div style='padding: 20px; color: red;'><strong>Error:</strong> {str(e)}</div>"

# Export app for uvicorn
__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
    )
