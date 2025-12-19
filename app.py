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
        
        # For old API with ConversationBufferMemory, we need to ensure memory is in correct state
        # Clear any accumulated state from extraction to avoid format conflicts
        if boq_processor.LC_OLD_API is True:
            # Old LangChain API (0.1.x): ConversationalRetrievalChain with ConversationBufferMemory
            # Reset memory to avoid format conflicts from previous extraction
            if hasattr(qa_chain, 'memory') and hasattr(qa_chain.memory, 'clear'):
                qa_chain.memory.clear()
                logger.debug("Cleared conversation memory before chat")
            
            result = qa_chain({"question": request.question})
        elif boq_processor.LC_OLD_API is False:
            # New LangChain API (1.x): create_retrieval_chain expects explicit 'input' and 'chat_history' keys
            result = qa_chain.invoke({"question": request.question, "chat_history": []})
        else:
            # Fallback: try to determine based on available methods
            logger.warning("LC_OLD_API not set, attempting auto-detection")
            try:
                # Try new API first (has invoke method)
                result = qa_chain.invoke({"question": request.question, "chat_history": []})
            except (TypeError, KeyError, AttributeError):
                # Fall back to old API (has __call__ method and memory-based chat_history)
                if hasattr(qa_chain, 'memory') and hasattr(qa_chain.memory, 'clear'):
                    qa_chain.memory.clear()
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