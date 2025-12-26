from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import boq_processor
from loguru import logger
import os
import shutil
from typing import Dict, Any

app = FastAPI(title="BOQ Chatbot API", description="API for extracting and querying BOQ from tender PDFs using RAG+CAG")

# Global variables for chain and vector store
vector_store = None
qa_chain = None
chunks = None

class ChatRequest(BaseModel):
    question: str

@app.post("/upload", summary="Upload PDF", description="Upload a PDF file to initialize the BOQ chatbot.")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    global vector_store, qa_chain
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file
    pdf_path = f"temp_{file.filename}"
    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing uploaded PDF: {file.filename}")
        global chunks
        chunks = boq_processor.load_and_process_pdf(pdf_path, filename=file.filename)
        vector_store = boq_processor.create_vector_store(chunks)
        qa_chain = boq_processor.setup_rag_chain(vector_store)
        
        # Use comprehensive extraction for complete BOQ coverage
        extracted_boq = boq_processor.extract_boq_comprehensive(chunks, vector_store)
        
        logger.info("PDF uploaded and processed successfully")
        
        # Return enriched response with processing metadata
        return {
            "status": "success",
            "message": "PDF uploaded and processed successfully",
            "file_name": file.filename,
            "processing_info": {
                "documents_loaded": len(chunks),
                "chunks_created": len(chunks),
                "vector_store_ready": vector_store is not None,
                "rag_chain_ready": qa_chain is not None
            },
            "extracted_boq": extracted_boq
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing PDF: {error_msg}")
        
        # Check if it's a rate limit error and provide specific guidance
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please try again later or upgrade your API plan. See https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        else:
            raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.post("/chat", summary="Chat with BOQ", description="Send a question about the uploaded BOQ PDF and get an answer.")
async def chat(request: ChatRequest) -> Dict[str, str]:
    if not qa_chain:
        logger.warning("Chat attempted without uploaded PDF")
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first using /upload")
    
    try:
        logger.info(f"Processing chat question: {request.question}")
        
        # Use old LangChain API (0.1.x) directly
        result = qa_chain({"question": request.question})
        
        logger.info("Chat response generated")
        return {"answer": result["answer"]}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in chat: {error_msg}")
        
        # Check if it's a rate limit error and provide specific guidance
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please try again later or upgrade your API plan. See https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/consistency", summary="Check Consistency", description="Run multiple BOQ extractions to check LLM output consistency.")
async def get_consistency() -> Dict[str, Any]:
    global chunks, vector_store
    if not chunks or not vector_store:
        logger.warning("Consistency check attempted without uploaded PDF")
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first using /upload")
    
    try:
        logger.info("Running consistency check")
        consistency_result = boq_processor.check_consistency(chunks, vector_store, runs=4)
        logger.info(f"Consistency check completed: {consistency_result}")
        return {"status": "success", "consistency": consistency_result}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in consistency check: {error_msg}")
        
        # Check if it's a rate limit error
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please try again later or upgrade your API plan. See https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        else:
            raise HTTPException(status_code=500, detail="Internal server error")