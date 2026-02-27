---
title: BOQ Tenders Agent
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BOQTenders - Bill of Quantities Extractor

A stateful agent system for extracting Bill of Quantities (BOQ) from tender documents and enabling document chat using LangGraph, LangChain, and Google Gemini LLM.

## 🏗️ Architecture

```
BOQTenders/
├── config/
│   └── settings.py         # Pydantic settings with all configurable parameters
├── core/
│   ├── agent.py            # LangGraph-based agent for workflow orchestration
│   ├── pdf_extractor.py    # PDF text extraction via HuggingFace API
│   ├── embeddings.py       # Text chunking and FAISS vector store
│   ├── llm.py              # Google Gemini LLM client wrapper
│   └── rag_chain.py        # RAG chain builder for document Q&A
├── services/
│   ├── boq_extractor.py    # BOQ extraction service with iterative consistency
│   ├── consistency.py      # Consistency checking service
│   ├── mongo_store.py      # MongoDB event logging
│   └── s3_utils.py         # AWS S3 file storage
├── api/
│   ├── routes.py           # FastAPI routes for /get_boq and /chat
│   └── schemas.py          # Pydantic request/response models
├── prompts/
│   ├── get_prompts.py      # Prompt loader
│   └── templates.yaml      # LLM prompt templates
├── app.py                  # FastAPI entry point
├── streamlit_app.py        # Streamlit UI entry point
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── requirements.txt        # Python dependencies
```

## ✨ Features

- **📄 PDF Processing**: Extract text from tender documents using HuggingFace API
- **🔍 BOQ Extraction**: Automatically identify and extract BOQ items with:
  - Item codes, descriptions, units, quantities
  - Unit prices, total amounts
  - Confidence scores for each item
  - Source page references
- **🔄 Consistency Checking**: Iterative extraction with multiple runs for accuracy
- **💬 Document Chat**: Ask questions about processed documents using RAG
- **🗂️ Stateful Agent**: LangGraph workflow for one-time extraction + multiple chats
- **📊 Logging & Storage**: MongoDB event logging and S3 file storage
- **🌐 Web UI**: Streamlit interface for easy document upload and interaction
- **🚀 API**: FastAPI backend for programmatic access

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- MongoDB (optional, for logging)
- AWS S3 (optional, for file storage)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BOQTenders
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   HF_API_TOKEN=your_huggingface_token
   LOG_LEVEL=DEBUG
   # Optional: MongoDB and S3 configs
   ```

### Usage

#### Streamlit UI
```bash
streamlit run streamlit_app.py
```
- Upload PDF, enter API key, process document for BOQ extraction.
- Chat with the document using the same API key.

#### FastAPI Backend
```bash
uvicorn app:app --reload
```

#### API Endpoints
- `POST /get_boq`: Extract BOQ from uploaded PDF
- `POST /chat`: Chat with processed document

## 📚 API Documentation

### Extract BOQ
```bash
curl -X POST "http://localhost:8000/get_boq" \
     -H "Content-Type: application/json" \
     -d '{
       "file": "base64_encoded_pdf",
       "api_key": "your_api_key",
       "runs": 2,
       "boq_mode": ["default"]
     }'
```

### Chat with Document
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "process_id": "session_id",
       "question": "What is the total quantity?",
       "api_key": "your_api_key"
     }'
```

## 🛠️ Development

### Project Structure
- `core/agent.py`: Main LangGraph agent with simplified workflow
- `api/routes.py`: FastAPI endpoints using agent
- `streamlit_app.py`: Web UI with session management
- `services/`: Business logic for extraction, consistency, storage

### Key Components
- **Agent Workflow**: Linear extraction graph + direct chat calls
- **State Management**: AgentState TypedDict for workflow state
- **Error Handling**: Graceful failures with logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
