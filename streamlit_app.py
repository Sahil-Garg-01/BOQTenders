import streamlit as st
import tempfile
import os
import boq_processor

st.set_page_config(page_title="BOQ Chatbot", page_icon="📄", layout="wide")

st.title("BOQ Agent")
st.markdown("Upload a tender PDF to extract the Bill of Quantities (BOQ) and chat with the document.")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "extracted_boq" not in st.session_state:
    st.session_state.extracted_boq = None

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Upload a PDF containing BOQ data.
    - Automatically extract the BOQ.
    - Ask questions about the document (Coming Soon).
    """)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("🔄 Processing PDF..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process
                chunks = boq_processor.load_and_process_pdf(tmp_path)
                vector_store = boq_processor.create_vector_store(chunks)
                qa_chain = boq_processor.setup_rag_chain(vector_store)
                
                # Use comprehensive extraction for complete BOQ coverage
                extracted_boq = boq_processor.extract_boq_comprehensive(chunks, vector_store)
                
                # Store in session
                st.session_state.qa_chain = qa_chain
                st.session_state.extracted_boq = extracted_boq
                
                st.success("✅ PDF processed successfully!")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    st.error(f"⚠️ API Rate Limit Exceeded: You've hit the daily quota limit. Please try again later or upgrade your API plan. See https://ai.google.dev/gemini-api/docs/rate-limits")
                else:
                    st.error(f"Error processing PDF: {error_msg}")
            finally:
                os.unlink(tmp_path)

# Display extracted BOQ
if st.session_state.extracted_boq:
    st.subheader("📊 Extracted Bill of Quantities")
    
    # Parse and display the extracted BOQ with better formatting
    boq_text = st.session_state.extracted_boq
    
    # Display as markdown for better rendering
    st.markdown(boq_text)
    
    st.divider()
    
    # Add a download button for the BOQ below
    st.download_button(
        label="📥 Download BOQ as Text",
        data=boq_text,
        file_name="BOQ_extracted.txt",
        mime="text/plain"
    )

# Chat interface - DISABLED (CAG/RAG to be fixed in next update)
# TODO: Re-enable chat functionality after fixing RAG chain issues
else:
    st.info("💡Chat feature coming soon...")