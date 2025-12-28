"""
BOQTenders Streamlit Application

Interactive web interface for BOQ extraction and document chat.

Usage:
    streamlit run streamlit_app_new.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import tempfile
import streamlit as st
from loguru import logger

from config.settings import settings
from core.pdf_extractor import PDFExtractor
from core.embeddings import EmbeddingService
from core.rag_chain import RAGChainBuilder
from services.boq_extractor import BOQExtractor
from services.consistency import ConsistencyChecker

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def initialize_services(api_key: str):
    """Initialize all services with API key (cached)."""
    if "services_initialized" not in st.session_state or st.session_state.get("current_api_key") != api_key:
        from core.llm import LLMClient
        llm_client = LLMClient(api_key=api_key)
        st.session_state.pdf_extractor = PDFExtractor()
        st.session_state.embedding_service = EmbeddingService()
        st.session_state.rag_builder = RAGChainBuilder(llm_client=llm_client)
        st.session_state.boq_extractor = BOQExtractor(llm_client=llm_client)
        st.session_state.consistency_checker = ConsistencyChecker(boq_extractor=st.session_state.boq_extractor)
        st.session_state.services_initialized = True
        st.session_state.current_api_key = api_key


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "boq_output": None,
        "qa_chain": None,
        "vector_store": None,
        "chunks": None,
        "chat_history": [],
        "document_loaded": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_pdf(uploaded_file) -> bool:
    """
    Process uploaded PDF file.
    
    Returns:
        True if processing succeeded, False otherwise.
    """
    try:
        with st.spinner("Processing PDF..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Extract text
                st.info("Extracting text from PDF...")
                text = st.session_state.pdf_extractor.extract_text(temp_path, filename=uploaded_file.name)
                
                if not text:
                    st.error("Could not extract text from PDF")
                    return False
                
                # Create embeddings
                st.info("Creating embeddings...")
                chunks = st.session_state.embedding_service.split_text(text)
                vector_store = st.session_state.embedding_service.create_vector_store(chunks)
                
                # Extract BOQ
                st.info("Extracting BOQ items...")
                boq_output = st.session_state.boq_extractor.extract(chunks, vector_store)
                
                # Build QA chain
                st.info("Building chat interface...")
                qa_chain = st.session_state.rag_builder.build(vector_store)
                
                # Store in session
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.boq_output = boq_output
                st.session_state.qa_chain = qa_chain
                st.session_state.document_loaded = True
                st.session_state.chat_history = []
                
                st.success(f"✅ Processed {len(chunks)} document chunks")
                return True
                
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {str(e)}")
        return False


def render_chat_interface():
    """Render the chat interface."""
    st.subheader("💬 Chat with Document")
    
    if not st.session_state.document_loaded:
        st.info("Please upload a PDF to enable chat")
        return
    
    # Chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain({"question": prompt})
                answer = response.get("answer", "I couldn't find an answer.")
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)


def render_boq_output():
    """Render the BOQ output."""
    st.subheader("📋 Extracted BOQ")
    
    if st.session_state.boq_output:
        st.markdown(st.session_state.boq_output)
        
        # Download button
        st.download_button(
            label="📥 Download BOQ as Markdown",
            data=st.session_state.boq_output,
            file_name="boq_output.md",
            mime="text/markdown"
        )
    else:
        st.info("Upload a PDF to see extracted BOQ items")


def render_consistency_check():
    """Render consistency check interface."""
    st.subheader("🔍 Consistency Check")
    
    if not st.session_state.document_loaded:
        st.info("Upload a PDF to run consistency checks")
        return
    
    runs = st.number_input(
        "Number of extraction runs",
        min_value=2,
        max_value=10,
        value=settings.consistency.default_runs,
        step=1
    )
    
    if st.button("Run Consistency Check"):
        with st.spinner(f"Running {runs} extraction passes..."):
            try:
                result = st.session_state.consistency_checker.check(
                    chunks=st.session_state.chunks,
                    vector_store=st.session_state.vector_store,
                    runs=runs
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Consistency Score", f"{result['consistency_score']:.1f}%")
                
                with col2:
                    st.metric("Avg Confidence", f"{result['avg_confidence']:.1f}%")
                
                with col3:
                    st.metric("Successful Runs", f"{result['successful_runs']}/{result['runs']}")
                
                if result['is_low_consistency']:
                    st.warning("⚠️ Low consistency detected. Results may vary.")
                else:
                    st.success("✅ Good consistency across extraction runs")
                    
            except Exception as e:
                logger.error(f"Consistency check error: {e}")
                st.error(f"Error: {str(e)}")


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title("📄 BOQ Extractor")
        st.markdown("---")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Generative AI API key",
            key="api_key_input"
        )
        
        if api_key:
            initialize_services(api_key)
        else:
            st.warning("Please enter your LLM API key to proceed.")
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a tender/BOQ document for extraction"
        )
        
        if uploaded_file and api_key:
            if st.button("🚀 Process Document"):
                process_pdf(uploaded_file)
        elif uploaded_file and not api_key:
            st.error("Please enter API key first.")
        
        st.markdown("---")
        
        # Clear session
        if st.button("🗑️ Clear Session"):
            for key in list(st.session_state.keys()):
                if key not in ["services_initialized", "current_api_key"]:
                    del st.session_state[key]
            initialize_session_state()
            st.success("Session cleared!")
            st.rerun()


def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title=settings.streamlit.page_title,
        page_icon=settings.streamlit.page_icon,
        layout=settings.streamlit.layout,
        initial_sidebar_state="expanded"
    )
    
    # Add CSS for sticky tabs
    st.markdown("""
        <style>
        /* Make tabs sticky at top */
        .stTabs [data-baseweb="tab-list"] {
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 999;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e6e6e6;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .stTabs [data-baseweb="tab-list"] {
                background-color: #0e1117;
                border-bottom: 1px solid #333;
            }
        }
        
        /* Streamlit dark theme */
        [data-theme="dark"] .stTabs [data-baseweb="tab-list"] {
            background-color: #0e1117;
            border-bottom: 1px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📋 BOQ Output", "💬 Chat", "🔍 Analysis"])
    
    with tab1:
        render_boq_output()
    
    with tab2:
        render_chat_interface()
    
    with tab3:
        render_consistency_check()


if __name__ == "__main__":
    main()
