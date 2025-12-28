---
title: BOQ Tenders Agent
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# BOQTenders - Bill of Quantities Extractor

A modular, production-ready system for extracting Bill of Quantities (BOQ) from tender documents using LangChain RAG and Gemini LLM.

## 🏗️ Architecture

```
BOQTenders/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Pydantic settings with all configurable parameters
├── core/                   # Core components
│   ├── __init__.py
│   ├── pdf_extractor.py    # PDF text extraction
│   ├── embeddings.py       # Text chunking and vector store
│   ├── llm.py              # LLM client wrapper
│   └── rag_chain.py        # RAG chain builder
├── services/               # Business logic services
│   ├── __init__.py
│   ├── boq_extractor.py    # BOQ extraction service
│   └── consistency.py      # Consistency checking service
├── api/                    # FastAPI endpoints
│   ├── __init__.py
│   ├── routes.py           # API routes
│   └── schemas.py          # Pydantic request/response models
├── prompts/                # LLM prompt templates
│   ├── __init__.py
│   └── templates.py        # All prompt templates
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── retry.py            # Retry decorator with exponential backoff
├── app_new.py              # FastAPI entry point
├── streamlit_app_new.py    # Streamlit UI entry point
├── .env.example            # Environment variables template
└── requirements_new.txt    # Python dependencies
```

## ✨ Features

- **📄 PDF Processing**: Extract text from tender documents
- **🔍 BOQ Extraction**: Automatically identify and extract BOQ items with:
  - Item codes, descriptions, units, quantities
  - Unit prices, total amounts
  - Confidence scores for each item
  - Source page references
- **💬 Document Chat**: Ask questions about the document using RAG
- **📊 Consistency Check**: Validate extraction reliability across multiple runs
- **🔧 Fully Configurable**: All parameters configurable via environment variables

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd BOQTenders
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements_new.txt
```

### 2. Configure Environment

```bash
# Copy example and edit with your API keys
cp .env.example .env
```

Required API keys:
- `GOOGLE_API_KEY`: For Gemini LLM
- `HUGGINGFACE_API_KEY`: For embeddings and PDF processing

### 3. Run the Application

**Streamlit UI:**
```bash
streamlit run streamlit_app_new.py
```

**FastAPI Server:**
```bash
python app_new.py
# Or: uvicorn app_new:app --reload
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF for processing |
| POST | `/chat` | Chat with document |
| POST | `/consistency` | Run consistency check |
| POST | `/clear` | Clear session |

### Example Usage

```python
import requests

# Upload PDF
with open("tender.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
print(response.json()["boq_output"])

# Chat with document
response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is the total steel quantity?"}
)
print(response.json()["answer"])
```

## ⚙️ Configuration

All settings can be configured via environment variables or `.env` file:

### LLM Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | Google API key (required) |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Gemini model name |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens |

### Embedding Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `EMBEDDING_CHUNK_SIZE` | `1000` | Text chunk size |
| `EMBEDDING_CHUNK_OVERLAP` | `200` | Chunk overlap |

### BOQ Extraction Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `BOQ_BATCH_SIZE` | `10` | Chunks per batch |
| `BOQ_MAX_PROMPT_LENGTH` | `15000` | Max prompt chars |
| `BOQ_PAGE_SEARCH_LENGTH` | `100` | Chars for page detection |

### API Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |
| `API_DEBUG` | `false` | Debug mode |

## 🧪 Consistency Checking

The consistency checker runs multiple extraction passes and calculates:
- **Consistency Score**: Average pairwise similarity (%)
- **Average Confidence**: Mean confidence across all items
- **Success Rate**: Proportion of successful extractions

```python
from services.consistency import ConsistencyChecker

checker = ConsistencyChecker()
result = checker.check(chunks, vector_store, runs=4)
print(f"Consistency: {result['consistency_score']}%")
```

## 🔧 Extending the System

### Adding a New Service

```python
# services/my_service.py
from config.settings import settings

class MyService:
    def __init__(self, param: str = None):
        self.param = param or settings.my_setting
    
    def process(self, data):
        # Implementation
        pass
```

### Adding a New API Endpoint

```python
# In api/routes.py
@app.post("/my-endpoint")
async def my_endpoint(request: MyRequest):
    service = MyService()
    return service.process(request.data)
```

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

## 🔄 Migration from Old Structure

If upgrading from the old monolithic structure:

1. Install new requirements: `pip install -r requirements_new.txt`
2. Copy your `.env` file or update with new variable names
3. Update imports in any custom code:
   ```python
   # Old
   from boq_processor import extract_boq_comprehensive
   
   # New
   from services.boq_extractor import BOQExtractor
   extractor = BOQExtractor()
   result = extractor.extract(chunks)
   ```

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
