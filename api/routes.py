"""
FastAPI routes for BOQ extraction API using LangGraph agent.
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
from core.agent import BOQAgent
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

# Global agent instance
_agent = None

# Global session state (for single-user API)
_session_state = {
    "qa_chain": None,
    "vector_store": None,
    "chunks": None,
    "api_key": None,
    "process_id": None,
    "boq_output": None,
    "consistency": None,
    "chat_history": [],
}

def get_agent():
    global _agent
    if _agent is None:
        _agent = BOQAgent()
    return _agent



@router.post(
    "/get_boq",
    response_model=GetBoqResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Documents"]
)
async def get_boq(file: UploadFile = File(...), api_key: str = Form(...), runs: int = Form(2), boq_mode: str = Form(None), specific_boq: str = Form(None)):
    """
    Get BOQ from a PDF file using LangGraph agent.
    
    - Processes PDF through agent workflow
    - Extracts BOQ items with iterative improvement
    - Computes consistency metrics
    - Sets up QA chain for chat
    
    Parameters:
    - file: PDF file to process
    - api_key: Google API key for LLM
    - runs: Number of extraction runs (1-5)
    - boq_mode: JSON list of modes ["default"] or ["specific BOQ"] or both
    - specific_boq: Specific BOQ name to extract (if "specific BOQ" in boq_mode)
    """
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if runs < 1 or runs > 5:
        raise HTTPException(status_code=400, detail="Runs must be between 1 and 5")
    
    # Parse boq_mode
    boq_mode_list = []
    if boq_mode:
        try:
            boq_mode_list = json.loads(boq_mode)
            if not isinstance(boq_mode_list, list):
                raise ValueError("boq_mode must be a list")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid boq_mode format, must be JSON list")
    
    try:
        logger.info(f'Processing file: {file.filename}')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Upload to S3
        process_id = str(uuid.uuid4())
        s3_key = f"uploads/{process_id}_{file.filename}"
        upload_success = upload_to_s3(temp_path, s3_key)
        input_url = generate_presigned_get_url(s3_key) if upload_success else None
        
        # Log created event
        insert_event({
            "id": process_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "get_boq",
            "status": "created",
            "filename": file.filename,
            "input_file_path": input_url,
        })
        
        # Process with agent
        agent = get_agent()
        initial_state = {
            "process_id": process_id,
            "api_key": api_key,
            "file_path": temp_path,
            "file_name": file.filename,
            "action": "extract_boq",
            "runs": runs,
            "boq_mode": boq_mode_list,
            "specific_boq": specific_boq,
            "question": None,
            "chat_history": [],
            "extracted_text": None,
            "chunks": None,
            "vector_store": None,
            "boq_output": None,
            "consistency": None,
            "qa_chain": None,
            "error": None,
        }
        result = agent.run(initial_state)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Store session for chat
        global _session_state
        _session_state = {
            "qa_chain": result.get("qa_chain"),
            "vector_store": result.get("vector_store"),
            "chunks": result.get("chunks"),
            "process_id": result["process_id"],
            "api_key": api_key,
            "chat_history": result.get("chat_history", []),
        }
        
        return GetBoqResponse(
            message="success",
            output=result["boq_output"],
            consistency_score=result["consistency"].get("consistency_score", 0),
            runs=result["consistency"].get("runs", runs),
            successful_runs=result["consistency"].get("successful_runs", 0),
            avg_confidence=result["consistency"].get("avg_confidence", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing BOQ extraction: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if 'temp_path' in locals():
            Path(temp_path).unlink(missing_ok=True)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Ask a question about the processed document using LangGraph agent.
    
    Requires a document to be processed first via /get_boq endpoint.
    """
    global _session_state
    
    if not _session_state.get("qa_chain"):
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please process a PDF first via /get_boq."
        )
    
    try:
        agent = get_agent()
        state = {
            "process_id": _session_state.get("process_id"),
            "api_key": _session_state.get("api_key"),
            "vector_store": _session_state.get("vector_store"),
            "qa_chain": _session_state.get("qa_chain"),
            "chunks": _session_state.get("chunks"),
            "action": "chat",
            "question": request.question,
            "chat_history": _session_state.get("chat_history", []),
            "runs": 1,
            "boq_mode": [],
            "specific_boq": None,
            "file_path": None,
            "file_name": None,
            "extracted_text": None,
            "boq_output": None,
            "consistency": None,
            "error": None,
        }
        result = agent.run(state)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update session with new chat history
        _session_state["chat_history"] = result.get("chat_history", [])
        
        answer = result.get("answer", "No answer generated.")
        
        return ChatResponse(answer=answer)
        
    except HTTPException:
        raise
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
        "api_key": None,
        "process_id": None,
        "boq_output": None,
        "consistency": None,
        "chat_history": [],
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
