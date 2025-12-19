import streamlit as st
import tempfile
import os
import boq_processor

st.set_page_config(page_title="BOQ Agent", page_icon="📄", layout="wide")

st.title("BOQ Agent")
st.markdown("Upload a tender PDF to extract the Bill of Quantities (BOQ) and chat with the document.")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "extracted_boq" not in st.session_state:
    st.session_state.extracted_boq = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Upload a PDF containing BOQ data.
    - Automatically extract the BOQ.
    - Ask questions about the document.
    """)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Generate BOQ"):
        with st.spinner("Generating BOQ..."):
            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process
                chunks = boq_processor.load_and_process_pdf(tmp_path, filename=uploaded_file.name)
                vector_store = boq_processor.create_vector_store(chunks)
                qa_chain = boq_processor.setup_rag_chain(vector_store)
                
                # Use comprehensive extraction for complete BOQ coverage
                extracted_boq = boq_processor.extract_boq_comprehensive(chunks, vector_store)
                
                # Store in session
                st.session_state.qa_chain = qa_chain
                st.session_state.extracted_boq = extracted_boq
                st.session_state.messages = []  # reset chat history

                st.success("✅ BOQ generated successfully!")
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

    # -------------------------------
    # Chat Interface
    # -------------------------------
    st.subheader("💬 Chat with your BOQ")

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the BOQ"):
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # OLD LangChain API (ConversationalRetrievalChain)
                    response = st.session_state.qa_chain(
                        {"question": prompt}
                    )

                    answer = response.get("answer", "No response generated.")
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Chat error: {e}")
