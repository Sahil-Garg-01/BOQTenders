"""
BOQTenders LangGraph Agent

Stateful agent for BOQ extraction and document chat using LangGraph.
"""
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path
import tempfile
from contextlib import nullcontext
from loguru import logger
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.pdf_extractor import PDFExtractor
from core.embeddings import EmbeddingService
from core.llm import LLMClient
from core.rag_chain import RAGChainBuilder
from services.boq_extractor import BOQExtractor
from services.consistency import ConsistencyChecker
from services.s3_utils import upload_to_s3, generate_presigned_get_url
from services.mongo_store import insert_event

# OpenTelemetry
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = None


class AgentState(TypedDict):
    """State for the BOQ extraction agent."""
    process_id: str
    api_key: str
    file_path: Optional[str]
    file_name: Optional[str]
    extracted_text: Optional[str]
    chunks: Optional[List[Document]]
    vector_store: Optional[FAISS]
    boq_output: Optional[str]
    consistency: Optional[Dict[str, Any]]
    qa_chain: Optional[Any]
    chat_history: List[Dict[str, str]]
    action: str  # "extract_boq", "chat"
    question: Optional[str]
    runs: int
    boq_mode: List[str]
    specific_boq: Optional[str]
    error: Optional[str]


class BOQAgent:
    """
    LangGraph-based agent for BOQ extraction and document processing.
    
    Handles PDF processing, BOQ extraction, chat, and consistency checking
    through a stateful workflow.
    """

    def __init__(self):
        """Initialize the agent with service instances."""
        self.pdf_extractor = PDFExtractor()
        self.embedding_service = EmbeddingService()
        self.extract_graph = self.create_extract_graph()

    def run(self, state: AgentState) -> AgentState:
        """Run the agent workflow with the given state."""
        with (tracer.start_as_current_span("agent_run") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("process_id", state.get("process_id", ""))
                span.set_attribute("action", state.get("action", ""))
            if state["action"] == "chat" and state.get("qa_chain"):
                return self._chat(state)
            elif state["action"] == "extract_boq":
                return self.extract_graph.invoke(state)
            else:
                state["error"] = "Invalid action or document not loaded"
                return state

    def create_extract_graph(self) -> StateGraph:
        """Create the extraction workflow."""
        workflow = StateGraph(AgentState)
        workflow.add_node("extract_text", self._extract_text)
        workflow.add_node("create_embeddings", self._create_embeddings)
        workflow.add_node("extract_boq", self._extract_boq)
        workflow.add_node("build_rag", self._build_rag)
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "create_embeddings")
        workflow.add_edge("create_embeddings", "extract_boq")
        workflow.add_edge("extract_boq", "build_rag")
        workflow.add_edge("build_rag", END)
        return workflow.compile()

    def _extract_text(self, state: AgentState) -> AgentState:
        """Extract text from PDF."""
        with (tracer.start_as_current_span("extract_text") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("file_name", state.get("file_name", ""))
            try:
                logger.info(f'Extracting text from {state["file_name"]}')

                text = self.pdf_extractor.extract_text(
                    state["file_path"],
                    filename=state["file_name"]
                )

                if not text:
                    raise ValueError("Could not extract text from PDF")

                # Log progress
                insert_event({
                    "id": state["process_id"],
                    "timestamp": self._get_timestamp(),
                    "service": "agent",
                    "current_step": "text_extraction",
                    "status": "completed",
                })

                return {**state, "extracted_text": text}

            except Exception as e:
                logger.error(f'Text extraction failed: {e}')
                return {**state, "error": str(e)}

    def _create_embeddings(self, state: AgentState) -> AgentState:
        """Create embeddings and vector store."""
        with (tracer.start_as_current_span("create_embeddings") if tracer else nullcontext()):
            try:
                logger.info('Creating embeddings and vector store')

                chunks = self.embedding_service.split_text(state["extracted_text"])
                vector_store = self.embedding_service.create_vector_store(chunks)

                # Log progress
                insert_event({
                    "id": state["process_id"],
                    "timestamp": self._get_timestamp(),
                    "service": "agent",
                    "current_step": "embedding_creation",
                    "status": "completed",
                })

                return {**state, "chunks": chunks, "vector_store": vector_store}

            except Exception as e:
                logger.error(f'Embedding creation failed: {e}')
                return {**state, "error": str(e)}

    def _extract_boq(self, state: AgentState) -> AgentState:
        """Extract BOQ from document."""
        with (tracer.start_as_current_span("extract_boq") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("runs", state.get("runs", 0))
            try:
                logger.info(f'Extracting BOQ with {state["runs"]} runs')

                llm_client = LLMClient(api_key=state["api_key"])
                boq_extractor = BOQExtractor(llm_client=llm_client)
                consistency_checker = ConsistencyChecker(boq_extractor=boq_extractor)

                final_boq, all_outputs = boq_extractor.extract_iterative(
                    state["chunks"],
                    state["vector_store"],
                    state["runs"],
                    state["boq_mode"],
                    state["specific_boq"]
                )

                consistency = consistency_checker.check_from_outputs(all_outputs)

                # Upload to S3
                self._upload_boq_to_s3(state["process_id"], final_boq, consistency)

                # Log completion
                insert_event({
                    "id": state["process_id"],
                    "timestamp": self._get_timestamp(),
                    "service": "agent",
                    "status": "completed",
                    "answer": final_boq,
                    "runs": state["runs"],
                })

                return {
                    **state,
                    "boq_output": final_boq,
                    "consistency": consistency
                }

            except Exception as e:
                logger.error(f'BOQ extraction failed: {e}')
                return {**state, "error": str(e)}

    def _build_rag(self, state: AgentState) -> AgentState:
        """Build RAG chain for chat."""
        with (tracer.start_as_current_span("build_rag") if tracer else nullcontext()):
            try:
                # Validate vector store exists before building RAG
                if state.get("vector_store") is None:
                    error_msg = "Cannot build RAG: No vector store available. Please extract BOQ from document first."
                    logger.error(error_msg)
                    return {**state, "error": error_msg}

                logger.info('Building RAG chain')

                llm_client = LLMClient(api_key=state["api_key"])
                rag_builder = RAGChainBuilder(llm_client=llm_client)
                qa_chain = rag_builder.build(state["vector_store"])

                return {**state, "qa_chain": qa_chain}

            except Exception as e:
                logger.error(f'RAG building failed: {e}')
                return {**state, "error": str(e)}

    def _chat(self, state: AgentState) -> AgentState:
        """Handle chat interaction."""
        with (tracer.start_as_current_span("chat") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("question_length", len(state.get("question", "")))
            try:
                logger.info(f'Processing chat question: {state["question"][:50]}...')

                # Sync memory with chat_history
                memory = state["qa_chain"].memory
                memory.chat_memory.clear()
                for msg in state["chat_history"]:
                    if msg["role"] == "user":
                        memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        memory.chat_memory.add_ai_message(msg["content"])

                # Log question
                insert_event({
                    "id": f"{state['process_id']}_chat",
                    "timestamp": self._get_timestamp(),
                    "service": "chat",
                    "event_type": "question",
                    "status": "created",
                    "question": state["question"],
                })

                try:
                    response = state["qa_chain"].invoke({"question": state["question"]})
                    answer = response.get("answer", "")
                    logger.info(f'Chat response received: {answer[:100]}...')
                except Exception as e:
                    logger.error(f'Chat invoke failed: {e}')
                    answer = f"Error: {str(e)}"

                # Log answer
                insert_event({
                    "id": f"{state['process_id']}_chat",
                    "timestamp": self._get_timestamp(),
                    "service": "chat",
                    "event_type": "answer",
                    "status": "completed",
                    "answer": answer,
                })

                # Update chat history
                chat_history = state["chat_history"] + [
                    {"role": "user", "content": state["question"]},
                    {"role": "assistant", "content": answer}
                ]

                state["answer"] = answer
                state["chat_history"] = chat_history
                return state

            except Exception as e:
                logger.error(f'Chat failed: {e}')
                return {**state, "error": str(e)}

    def _upload_boq_to_s3(self, process_id: str, boq_output: str, consistency: Dict) -> None:
        """Upload BOQ results to S3."""
        try:
            s3_data = {
                "output": boq_output,
                "consistency_score": consistency.get("consistency_score"),
                "runs": consistency.get("runs"),
                "successful_runs": consistency.get("successful_runs"),
                "avg_confidence": consistency.get("avg_confidence")
            }
            import json
            import io
            json_bytes = json.dumps(s3_data, ensure_ascii=False, indent=2).encode('utf-8')
            output_s3_key = f"outputs/boq_{process_id}.json"
            with io.BytesIO(json_bytes) as f:
                upload_to_s3(f, output_s3_key)
        except Exception as e:
            logger.warning(f'S3 upload failed: {e}')

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def process_document(
        self,
        file_path: str,
        file_name: str,
        api_key: str,
        runs: int = 2,
        boq_mode: List[str] = None,
        specific_boq: Optional[str] = None,
        action: str = "extract_boq"
    ) -> Dict[str, Any]:
        """
        Process a document through the agent workflow.
        
        Args:
            file_path: Path to the PDF file
            file_name: Name of the file
            api_key: Google API key
            runs: Number of extraction runs
            boq_mode: BOQ extraction modes
            specific_boq: Specific BOQ to extract
            action: Action to perform ("extract_boq", "chat", etc.)
            
        Returns:
            Processing results
        """
        import uuid

        initial_state: AgentState = {
            "process_id": str(uuid.uuid4()),
            "api_key": api_key,
            "file_path": file_path,
            "file_name": file_name,
            "extracted_text": None,
            "chunks": None,
            "vector_store": None,
            "boq_output": None,
            "consistency": None,
            "qa_chain": None,
            "chat_history": [],
            "action": action,
            "question": None,
            "runs": runs,
            "boq_mode": boq_mode or ["default"],
            "specific_boq": specific_boq,
            "error": None,
        }

        final_state = self.run(initial_state)

        return {
            "process_id": final_state["process_id"],
            "boq_output": final_state.get("boq_output"),
            "consistency": final_state.get("consistency"),
            "qa_chain": final_state.get("qa_chain"),
            "chat_history": final_state.get("chat_history", []),
            "error": final_state.get("error"),
        }

    def chat_with_document(
        self,
        process_id: str,
        question: str,
        qa_chain: Any,
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Chat with a processed document.
        
        Args:
            process_id: Process ID
            question: User question
            qa_chain: RAG chain
            chat_history: Existing chat history
            
        Returns:
            Answer
        """
        state: AgentState = {
            "process_id": process_id,
            "api_key": "",
            "file_path": None,
            "file_name": None,
            "extracted_text": None,
            "chunks": None,
            "vector_store": None,
            "boq_output": None,
            "consistency": None,
            "qa_chain": qa_chain,
            "chat_history": chat_history or [],
            "action": "chat",
            "question": question,
            "runs": 1,
            "boq_mode": [],
            "specific_boq": None,
            "error": None,
        }

        final_state = self.run(state)

        return final_state["chat_history"][-1]["content"] if final_state["chat_history"] else ""