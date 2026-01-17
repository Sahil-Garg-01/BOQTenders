"""
FastAPI routes for BOQ extraction API.
"""
import io
import json
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from loguru import logger

from config.settings import settings
from core.pdf_extractor import PDFExtractor
from core.embeddings import EmbeddingService
from core.rag_chain import RAGChainBuilder
from services.boq_extractor import BOQExtractor
from services.consistency import ConsistencyChecker
from services.s3_utils import upload_to_s3, generate_presigned_get_url
from services.mongo_store import insert_event, list_latest_events_for_service, get_latest_event_for_process
from api.schemas import (
    ChatRequest,
    ChatResponse,
    GetBoqResponse,
    ErrorResponse,
)


# Create API router
router = APIRouter()

# Global state for session data
_session_state = {
    "qa_chain": None,
    "vector_store": None,
    "chunks": None,
    "api_key": None,
}

# Services will be initialized lazily to avoid startup timeout
_pdf_extractor = None
_embedding_service = None

def get_pdf_extractor():
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service



@router.post(
    "/get_boq",
    response_model=GetBoqResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Documents"]
)
async def get_boq(file: UploadFile = File(...), api_key: str = Form(...), runs: int = Form(2)):
    """
    Get BOQ from a PDF file with iterative improvement and consistency checking.
    
    - Extracts text from PDF
    - Creates embeddings and vector store
    - Extracts BOQ items iteratively (runs 1-5)
    - Computes consistency metrics
    - Sets up QA chain for chat
    """
    global _session_state
    
    # Generate unique process ID
    process_id = str(uuid.uuid4())
    service = "get_boq"
    created_at = datetime.now(timezone.utc).isoformat()
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if runs < 1 or runs > 5:
        raise HTTPException(status_code=400, detail="Runs must be between 1 and 5")
    
    try:
        logger.info(f'Processing file: {file.filename}')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Upload to S3
        s3_key = f"uploads/{process_id}_{file.filename}"
        upload_success = upload_to_s3(temp_path, s3_key)
        input_url = generate_presigned_get_url(s3_key) if upload_success else None
        
        # Log created event
        insert_event({
            "id": process_id,
            "timestamp": created_at,
            "service": service,
            "status": "created",
            "filename": file.filename,
            "input_file_path": input_url,
        })
        
        try:
            # Extract text from PDF
            logger.info('Extracting text from PDF...')
            text = get_pdf_extractor().extract_text(temp_path, filename=file.filename)
            
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF"
                )
            
            # Progress update
            insert_event({
                "id": process_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": service,
                "current_step": "text_extraction",
                "status": "in_progress",
            })
            
            # Create LLM client with user API key
            from core.llm import LLMClient
            llm_client = LLMClient(api_key=api_key)
            
            # Create services with the LLM client
            boq_extractor_with_key = BOQExtractor(llm_client=llm_client)
            rag_builder_with_key = RAGChainBuilder(llm_client=llm_client)
            
            # Create chunks and vector store
            logger.info('Creating embeddings...')
            embedding_svc = get_embedding_service()
            chunks = embedding_svc.split_text(text)
            vector_store = embedding_svc.create_vector_store(chunks)
            
            # Progress update
            insert_event({
                "id": process_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": service,
                "current_step": "embedding_creation",
                "status": "in_progress",
            })
            
            # Extract BOQ
            logger.info('Extracting BOQ...')
            final_boq, all_outputs = boq_extractor_with_key.extract_iterative(chunks, vector_store, runs)
            consistency_checker_with_key = ConsistencyChecker(boq_extractor=boq_extractor_with_key)
            consistency = consistency_checker_with_key.check_from_outputs(all_outputs)
            boq_output = final_boq
            
            # Upload BOQ output to S3
            s3_data = {
                "output": boq_output,
                "consistency_score": consistency.get("consistency_score"),
                "runs": consistency.get("runs"),
                "successful_runs": consistency.get("successful_runs"),
                "avg_confidence": consistency.get("avg_confidence")
            }
            json_bytes = json.dumps(s3_data, ensure_ascii=False, indent=2).encode('utf-8')
            output_s3_key = f"outputs/boq_{process_id}.json"
            with io.BytesIO(json_bytes) as f:
                upload_to_s3(f, output_s3_key)
            
            # Progress update
            insert_event({
                "id": process_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": service,
                "current_step": "boq_extraction",
                "status": "in_progress",
            })
            
            # Build QA chain
            logger.info('Building QA chain...')
            qa_chain = rag_builder_with_key.build(vector_store)
            
            # Store in session state
            _session_state["qa_chain"] = qa_chain
            _session_state["vector_store"] = vector_store
            _session_state["chunks"] = chunks
            _session_state["api_key"] = api_key
            
            logger.info(f'BOQ extraction completed: {len(chunks)} chunks created')
            
            # Completed event
            finished_at = datetime.now(timezone.utc).isoformat()
            insert_event({
                "id": process_id,
                "timestamp": finished_at,
                "service": service,
                "status": "completed",
                "answer": boq_output,
                "runs": runs,
            })
            
            return GetBoqResponse(
                message="success",
                output=boq_output,
                consistency_score=consistency.get("consistency_score"),
                runs=consistency.get("runs"),
                successful_runs=consistency.get("successful_runs"),
                avg_confidence=consistency.get("avg_confidence")
            )
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing BOQ extraction: {e}')
        # Failed event
        insert_event({
            "id": process_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Ask a question about the processed document.
    
    Requires a document to be processed first via /get_boq endpoint.
    """
    global _session_state
    
    if not _session_state.get("qa_chain"):
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please process a PDF first via /get_boq."
        )
    
    try:
        logger.info(f'Processing chat question: {request.question[:50]}...')
        
        # Log question event
        chat_process_id = str(uuid.uuid4())
        insert_event({
            "id": chat_process_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "chat",
            "event_type": "question",
            "status": "created",
            "question": request.question,
        })
        
        qa_chain = _session_state["qa_chain"]
        
        # Get response from QA chain (using old LangChain API)
        response = qa_chain({"question": request.question})
        
        answer = response.get("answer", "")
        
        # Upload chat data to S3
        chat_data = {"question": request.question, "answer": answer}
        json_bytes = json.dumps(chat_data, ensure_ascii=False, indent=2).encode('utf-8')
        chat_s3_key = f"chats/chat_{uuid.uuid4()}.json"
        with io.BytesIO(json_bytes) as f:
            upload_to_s3(f, chat_s3_key)
        
        # Log answer event
        insert_event({
            "id": chat_process_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "chat",
            "event_type": "answer",
            "status": "completed",
            "answer": answer,
        })
        
        logger.info('Chat response generated')
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error(f'Error processing chat: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
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


@router.get("/processes", tags=["Processes"])
async def list_processes(limit: int = 5, service: Optional[str] = None):

    if service:
        records = list_latest_events_for_service(service, limit=limit)
        services = [service]
    else:
        # Combine from all services
        records = []
        services = ["get_boq", "chat"]
        for svc in services:
            try:
                records.extend(list_latest_events_for_service(svc, limit=limit))
            except Exception:
                continue
        # Sort by timestamp descending and limit
        records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    
    return {
        "services": services,
        "count": len(records),
        "records": records
    }


@router.get("/processes/{process_id}", tags=["Processes"])
async def get_process(process_id: str):
    """
    Get details of a specific process.
    """
    rec = get_latest_event_for_process(process_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Process not found")
    return rec
