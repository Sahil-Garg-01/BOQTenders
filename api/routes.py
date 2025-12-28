"""
FastAPI routes for BOQ extraction API.
"""
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings
from core.pdf_extractor import PDFExtractor
from core.embeddings import EmbeddingService
from core.rag_chain import RAGChainBuilder
from services.boq_extractor import BOQExtractor
from services.consistency import ConsistencyChecker
from api.schemas import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    ConsistencyResponse,
    ErrorResponse,
)


# Initialize FastAPI app
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

# Global state for session data
_session_state = {
    "qa_chain": None,
    "vector_store": None,
    "chunks": None,
}

# Initialize services
pdf_extractor = PDFExtractor()
embedding_service = EmbeddingService()

# Expose router for external use
router = app.router


@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Documents"]
)
async def upload_pdf(file: UploadFile = File(...), api_key: str = Form(...)):
    """
    Upload a PDF file for BOQ extraction.
    
    - Extracts text from PDF
    - Creates embeddings and vector store
    - Extracts BOQ items
    - Sets up QA chain for chat
    """
    global _session_state
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        logger.info(f'Processing uploaded file: {file.filename}')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Extract text from PDF
            logger.info('Extracting text from PDF...')
            text = pdf_extractor.extract_text(temp_path, filename=file.filename)
            
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF"
                )
            
            # Create LLM client with user API key
            from core.llm import LLMClient
            llm_client = LLMClient(api_key=api_key)
            
            # Create services with the LLM client
            boq_extractor_with_key = BOQExtractor(llm_client=llm_client)
            rag_builder_with_key = RAGChainBuilder(llm_client=llm_client)
            
            # Create chunks and vector store
            logger.info('Creating embeddings...')
            chunks = embedding_service.split_text(text)
            vector_store = embedding_service.create_vector_store(chunks)
            
            # Extract BOQ
            logger.info('Extracting BOQ...')
            boq_output = boq_extractor_with_key.extract(chunks, vector_store)
            
            # Build QA chain
            logger.info('Building QA chain...')
            qa_chain = rag_builder_with_key.build(vector_store)
            
            # Store in session state
            _session_state["qa_chain"] = qa_chain
            _session_state["vector_store"] = vector_store
            _session_state["chunks"] = chunks
            _session_state["api_key"] = api_key
            
            logger.info(f'Upload completed: {len(chunks)} chunks created')
            
            return UploadResponse(
                message="success",
                output=boq_output
            )
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing upload: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Ask a question about the uploaded document.
    
    Requires a document to be uploaded first via /upload endpoint.
    """
    global _session_state
    
    if not _session_state.get("qa_chain"):
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a PDF first."
        )
    
    try:
        logger.info(f'Processing chat question: {request.question[:50]}...')
        
        qa_chain = _session_state["qa_chain"]
        
        # Get response from QA chain (using old LangChain API)
        response = qa_chain({"question": request.question})
        
        answer = response.get("answer", "")
        
        logger.info('Chat response generated')
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error(f'Error processing chat: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/consistency",
    response_model=ConsistencyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"]
)
async def check_consistency(runs: int = 4):
    """
    Check extraction consistency by running multiple extractions.
    
    Requires a document to be uploaded first via /upload endpoint.
    
    Args:
        runs: Number of extraction runs (default: 4)
    """
    global _session_state
    
    if not _session_state.get("chunks"):
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a PDF first."
        )
    
    if runs < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 runs required for consistency check"
        )
    
    if runs > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 runs allowed"
        )
    
    try:
        logger.info(f'Running consistency check with {runs} runs')
        
        # Create BOQ extractor with stored API key
        from core.llm import LLMClient
        from services.boq_extractor import BOQExtractor
        llm_client = LLMClient(api_key=_session_state.get("api_key"))
        boq_extractor_with_key = BOQExtractor(llm_client=llm_client)
        consistency_checker_with_key = ConsistencyChecker(boq_extractor=boq_extractor_with_key)
        
        result = consistency_checker_with_key.check(
            chunks=_session_state["chunks"],
            vector_store=_session_state["vector_store"],
            runs=runs
        )
        
        return ConsistencyResponse(
            consistency_score=result.get("consistency_score"),
            successful_runs=result.get("successful_runs"),
            avg_confidence=result.get("avg_confidence"),
            is_low_consistency=result.get("is_low_consistency")
        )
        
    except Exception as e:
        logger.error(f'Error in consistency check: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/clear",
    tags=["Session"]
)
async def clear_session():
    """Clear the current session state."""
    global _session_state
    
    _session_state = {
        "qa_chain": None,
        "vector_store": None,
        "chunks": None,
    }
    
    logger.info('Session cleared')
    return {"message": "Session cleared"}
