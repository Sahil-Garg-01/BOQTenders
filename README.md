---
title: BOQ Tenders Agent
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
custom_headers:
  health_route: /health
---

# BOQTenders - Bill of Quantities Extractor

A modular, production-ready system for extracting Bill of Quantities (BOQ) from tender documents using LangChain RAG and Google Gemini LLM.

## 🏗️ Architecture

```
BOQTenders/
├── config/
│   └── settings.py         # Pydantic settings with all configurable parameters
├── core/
│   ├── pdf_extractor.py    # PDF text extraction via HuggingFace API
│   ├── embeddings.py       # Text chunking and FAISS vector store
│   ├── llm.py              # Google Gemini LLM client wrapper
│   └── rag_chain.py        # RAG chain builder for document Q&A
├── services/
│   ├── boq_extractor.py    # BOQ extraction service
│   └── consistency.py      # Consistency checking service
├── api/
│   ├── routes.py           # FastAPI routes
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
- **💬 Document Chat**: Ask questions about the document using RAG
- **📊 Consistency Check**: Validate extraction reliability across multiple runs
- **🔑 User-Provided API Key**: Google API key is provided by users at runtime (not stored)
- **🐳 Docker Ready**: Deployable to Hugging Face Spaces or any Docker environment

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Live Demo)

Visit: [BOQ Tenders Agent Space](https://huggingface.co/spaces/point9/BOQ_of_Tenders_Agent)

1. Enter your Google Generative AI API key in the sidebar
2. Upload a PDF tender document
3. View extracted BOQ and chat with your document

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/Sahil-Garg-01/BOQTenders.git
cd BOQTenders

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker

```bash
# Build and run
docker build -t boqtenders .
docker run -p 8000:8000 -p 8501:8501 -e HF_API_TOKEN=your_token boqtenders
```

## 🖥️ Running the Application

**Streamlit UI (Recommended):**
```bash
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

**FastAPI Server:**
```bash
uvicorn app:app --reload
```
Access API docs at: http://localhost:8000/docs

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF for processing (requires `api_key` form field) |
| POST | `/chat` | Chat with document |
| POST | `/consistency` | Run consistency check |
| GET | `/clear` | Clear session |

### Example Usage

```python
import requests

# Upload PDF with API key
with open("tender.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f},
        data={"api_key": "your-google-api-key"}
    )
print(response.json()["output"])

# Chat with document
response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is the total steel quantity?"}
)
print(response.json()["answer"])
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_API_TOKEN` | Yes | HuggingFace API token for PDF extraction |
| `GOOGLE_API_KEY` | No | Optional default (users provide at runtime) |

### LLM Settings (via config/settings.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `model_name` | `gemini-2.5-flash-lite` | Google Gemini model |
| `temperature` | `0.0` | Generation temperature |
| `max_output_tokens` | `8192` | Max output tokens |

### Embedding Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `chunk_size` | `1000` | Text chunk size |
| `chunk_overlap` | `500` | Chunk overlap |

## 📝 BOQ Output Format

Extracted BOQ is returned in markdown format:

```markdown
## DOCUMENT SUMMARY
Project: XYZ Construction
Date: 2024-01-15
...

## DETAILED BILL OF QUANTITIES
**Total Items Found:** 25

| Item Code | Description | Unit | Quantity | Unit Price | Total | Confidence | Source |
|-----------|-------------|------|----------|------------|-------|------------|--------|
| A001 | Steel reinforcement | MT | 500 | 85000 | 42500000 | 92% | Page 5 |
...
```

## 🧪 Consistency Checking

Run multiple extraction passes to validate reliability:

```python
# Via API
response = requests.post("http://localhost:8000/consistency?runs=4")
print(f"Consistency: {response.json()['consistency_score']}%")
```

Results include:
- **Consistency Score**: Average pairwise similarity (%)
- **Average Confidence**: Mean confidence across all items
- **Success Rate**: Proportion of successful extractions

## 🔐 Security

- **Google API Key**: Provided by users at runtime, never stored on server
- **HF API Token**: Set as environment variable/secret (for PDF extraction)
- **No data persistence**: Documents are processed in memory only

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
