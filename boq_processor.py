import os
from typing import List
import dotenv
from pydantic_settings import BaseSettings
from loguru import logger
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import importlib
import time
from functools import wraps
import re

# Will be set at runtime by setup_rag_chain to either True (old API) or False (new API)
LC_OLD_API = None

from langchain_core.documents import Document

# Load environment variables
dotenv.load_dotenv()

def retry_with_exponential_backoff(max_retries=3, initial_delay=2, backoff_factor=2):
    """Decorator for retry logic with exponential backoff for API rate limits.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # Check if it's a rate limit error (429 or RESOURCE_EXHAUSTED)
                    is_rate_limit = ("429" in error_str or 
                                    "RESOURCE_EXHAUSTED" in error_str or 
                                    "quota" in error_str.lower())
                    
                    if is_rate_limit and attempt < max_retries:
                        delay = initial_delay * (backoff_factor ** attempt)
                        logger.warning(f"Rate limit encountered. Retry {attempt + 1}/{max_retries} in {delay}s: {error_str}")
                        time.sleep(delay)
                    else:
                        # Not a rate limit error or last retry - raise immediately
                        if is_rate_limit and attempt == max_retries:
                            logger.error(f"Rate limit exhausted after {max_retries} retries")
                        raise
            
            # This should not be reached, but in case it is
            raise last_exception
        return wrapper
    return decorator

class Settings(BaseSettings):
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()

def load_and_process_pdf(pdf_path: str) -> List[Document]:
    try:
        logger.info(f"Loading PDF from {pdf_path}")
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error loading and processing PDF: {e}")
        raise

def create_vector_store(chunks: List[Document]) -> FAISS:
    try:
        logger.info("Creating embeddings and vector store")
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def setup_rag_chain(vector_store: FAISS):
    """Create and return a retrieval-augmented chain compatible with installed LangChain.

    This function detects the available LangChain API at runtime and builds a chain
    using the old `ConversationalRetrievalChain` (0.1.x) or the newer 1.x APIs.
    """
    global LC_OLD_API
    logger.info("Setting up RAG chain")
    llm = GoogleGenerativeAI(model=settings.model_name, temperature=settings.temperature)

    # Try old API first
    try:
        conv_module = importlib.import_module("langchain_classic.chains")
        # If import succeeds and has ConversationalRetrievalChain, use old API
        if hasattr(conv_module, "ConversationalRetrievalChain"):
            LC_OLD_API = True
            from langchain_classic.chains import ConversationalRetrievalChain
            from langchain_classic.memory import ConversationBufferMemory
            from langchain_core.prompts import PromptTemplate

            try:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                qa_template = """Use the following pieces of context to answer the question at the end.
    If the question is about extracting BOQ, provide a structured list of items including descriptions, quantities, units, rates, and amounts.
    If no BOQ is found, say so.

    {context}

    Question: {question}
    Answer:"""
                qa_prompt = PromptTemplate.from_template(qa_template)
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": qa_prompt},
                )
                logger.info("RAG chain set up successfully (old API)")
                return qa_chain
            except Exception as e:
                logger.error(f"Error building chain with old LangChain API: {e}")
                raise
    except Exception:
        # Old API not available, try new API imports
        pass

    # Try new API (LangChain 1.x)
    try:
        create_retrieval_chain = importlib.import_module("langchain.chains").create_retrieval_chain
        create_history_aware_retriever = importlib.import_module("langchain.chains").create_history_aware_retriever
        create_stuff_documents_chain = importlib.import_module("langchain.chains").create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        LC_OLD_API = False

        contextualize_q_system_prompt = (
            """Given a chat history and the latest user question which might reference context in the chat history, """
            """formulate a standalone question which can be understood without the chat history. Do not answer the question, """
            """just reformulate it if needed and otherwise return it as is."""
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, vector_store.as_retriever(), contextualize_q_prompt
        )

        qa_system_prompt = """Use the following pieces of context to answer the question at the end.
    If the question is about extracting BOQ, provide a structured list of items including descriptions, quantities, units, rates, and amounts.
    If no BOQ is found, say so.

    {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info("RAG chain set up successfully (new API)")
        return rag_chain
    except Exception as e:
        logger.error(f"Could not import a compatible LangChain API: {e}")
        raise RuntimeError(
            "Unsupported LangChain version installed. Install either a 0.1.x series LangChain that provides ConversationalRetrievalChain, "
            "or a 1.x series with create_retrieval_chain. See project README for recommended versions."
        ) from e

@retry_with_exponential_backoff(max_retries=3, initial_delay=2)
def extract_boq(qa_chain) -> str:
    """Legacy function - kept only for backward compatibility with chat interface.
    
    DO NOT USE for BOQ extraction. Use extract_boq_comprehensive() instead.
    """
    raise NotImplementedError(
        "extract_boq() is deprecated. Use extract_boq_comprehensive(chunks, vector_store) instead."
    )


@retry_with_exponential_backoff(max_retries=3, initial_delay=2)
def extract_boq_comprehensive(chunks: List[Document], vector_store: FAISS = None) -> str:
    """Comprehensive BOQ extraction using batched processing.
    
    This processes chunks in batches to minimize API calls while ensuring complete coverage.
    
    Args:
        chunks: All document chunks from the PDF
        vector_store: Optional vector store for metadata extraction
    
    Returns:
        Formatted BOQ string with all items found
    """
    try:
        logger.info(f"Starting comprehensive BOQ extraction from {len(chunks)} chunks")
        
        # Initialize LLM
        llm = GoogleGenerativeAI(model=settings.model_name, temperature=settings.temperature)
        
        # Step 1: Extract document metadata from first few chunks
        logger.info("Extracting document metadata")
        metadata_text = "\n\n".join([chunk.page_content for chunk in chunks[:3]])
        
        metadata_prompt = f"""Analyze this tender document excerpt and extract:
1. Document Type (e.g., Civil BOQ, Electrical BOQ, etc.)
2. Project Name
3. Any other relevant project information

Document excerpt:
{metadata_text[:2000]}

Provide a brief summary."""
        
        metadata_result = llm.invoke(metadata_prompt)
        
        # Note: This metadata call is separate but necessary for document context
        # Total API usage: 1 metadata + batches = typically 11 calls max, still under 20/day limit
        
        # Step 2: Batch chunks for efficient processing
        logger.info("Batching chunks for efficient API processing")
        batch_size = 24  # Process 24 chunks per API call (240 chunks = 10 API calls)
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Created {len(batches)} batches from {len(chunks)} chunks (batch_size={batch_size}) - ~{len(batches)} API calls total")
        
        boq_items = []
        
        # Step 3: Extract BOQ items from each batch
        extraction_prompt_template = """Analyze this text and extract ONLY Bill of Quantities (BOQ) line items if present.

Look for structured data with:
- Item numbers or codes
- Descriptions of work/materials
- Quantities
- Units (Nos, Sqm, Cum, m, etc.)
- Rates/Unit prices
- Total amounts

If you find BOQ items, return them in this EXACT format (pipe-separated):
ITEM_CODE|DESCRIPTION|QUANTITY|UNIT|RATE|AMOUNT

Rules for columns:
- If an entire column (for example, `RATE` or `AMOUNT`) has no values anywhere in the document, omit that column from the output entirely (lines will have fewer fields).
- Otherwise always return all six fields; for any individual missing value return "NA" for that field so the model still returns 6 fields per line.

Return multiple items on separate lines. If NO BOQ items are found, return: "NO_BOQ_ITEMS"

Text to analyze:
{text}

Extract only actual BOQ line items, not headers or table structure."""
        
        for batch_num, batch_chunks in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)}")
            
            # Combine all chunks in this batch
            chunk_texts = [chunk.page_content for chunk in batch_chunks]
            batch_text = "\n\n".join(chunk_texts)
            # compute start/end offsets for each chunk in batch_text for source mapping
            offsets = []
            pos = 0
            for ci, txt in enumerate(chunk_texts):
                start = pos
                end = start + len(txt)
                md = getattr(batch_chunks[ci], 'metadata', {}) or {}
                offsets.append((start, end, md, ci))
                pos = end + 2
            
            # Extract BOQ items from this batch
            # Allow larger batches but cap to avoid overly long prompts
            max_chars = 30000
            prompt_text = batch_text if len(batch_text) <= max_chars else batch_text[:max_chars]
            prompt = extraction_prompt_template.format(text=prompt_text)
            
            try:
                result = llm.invoke(prompt)
                
                # Parse the result
                if result and "NO_BOQ_ITEMS" not in str(result):
                    # Split by lines and filter valid items
                    lines = str(result).strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line or '|' not in line:
                            continue

                        parts = [p.strip() for p in line.split('|')]
                        # Ensure at least 6 fields; if fewer, pad with 'NA'
                        if len(parts) < 6:
                            parts += ['NA'] * (6 - len(parts))

                        # Try to map this extracted line back to a source chunk (page) and append inline tag
                        source_tag = None
                        try:
                            snippet = parts[1].strip()[:60]
                            if snippet:
                                idx_found = batch_text.find(snippet)
                                if idx_found != -1:
                                    for start, end, md, cidx in offsets:
                                        if start <= idx_found <= end:
                                            page = md.get('page') or md.get('page_number') or md.get('pageIndex') or md.get('pageindex')
                                            if page is None:
                                                page = cidx + 1
                                            source_tag = f"p.{page}"
                                            break
                        except Exception:
                            source_tag = None

                        if source_tag and parts[1]:
                            if '(' not in parts[1]:
                                parts[1] = f"{parts[1]} ({source_tag})"

                        normalized_line = '|'.join(parts[:6])
                        boq_items.append(normalized_line)
                        logger.debug(f"Found BOQ item: {normalized_line[:120]}...")
            except Exception as e:
                logger.warning(f"Error processing batch {batch_num}: {e}")
                continue
        
        logger.info(f"Found {len(boq_items)} BOQ items across all {len(batches)} batches")
        
        # Step 4: Deduplicate items (in case of overlap between batches)
        unique_items = list(dict.fromkeys(boq_items))  # Preserves order, removes duplicates
        logger.info(f"After deduplication: {len(unique_items)} unique items")
        
        # Step 5: Format the final output
        if not unique_items:
            return f"""## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
No BOQ items were found in this document. This may not be a Bill of Quantities document, or the format may not be recognized."""
        
        # Determine which columns actually contain values across all items
        col_headers = ["Item No/Code", "Description", "Quantity", "Unit", "Rate", "Amount"]
        cols_present = [False] * 6
        normalized_items = []
        for item in unique_items:
            parts = [p.strip() for p in item.split('|')]
            if len(parts) < 6:
                parts += ['NA'] * (6 - len(parts))
            normalized_items.append(parts[:6])
            for i in range(6):
                if parts[i] and parts[i].upper() != 'NA':
                    cols_present[i] = True

        # If a column is entirely empty (all NA), omit it from the table
        col_indices = [i for i, present in enumerate(cols_present) if present]
        # Always include Item No/Code and Description even if missing (fallback)
        if 0 not in col_indices:
            col_indices.insert(0, 0)
        if 1 not in col_indices:
            col_indices.insert(1, 1)

        # Build table header dynamically
        header_row = "| " + " | ".join([col_headers[i] for i in col_indices]) + " |\n"
        sep_row = "|" + "|".join(["-" * (len(col_headers[i]) + 2) for i in col_indices]) + "|\n"

        formatted_boq = f"""## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
**Total Items Found:** {len(unique_items)}

    {header_row}{sep_row}"""

        for parts in normalized_items:
            # Truncate long descriptions
            parts[1] = parts[1][:80] if len(parts[1]) > 80 else parts[1]
            row_vals = [parts[i] for i in col_indices]
            formatted_boq += "| " + " | ".join(row_vals) + " |\n"
        
        # Calculate totals if possible
        total_amount = 0
        valid_amounts = 0
        for item in unique_items:
            parts = item.split('|')
            if len(parts) >= 6:
                try:
                    amount_str = parts[5].replace(',', '').replace('₹', '').strip()
                    amount = float(amount_str)
                    total_amount += amount
                    valid_amounts += 1
                except:
                    pass
        
        formatted_boq += f"""

## SUMMARY
- **Total Items:** {len(unique_items)}
"""

        # Normalize markdown: unify newlines, strip leading spaces, ensure proper table separation
        try:
            s = formatted_boq.replace('\r\n', '\n').replace('\r', '\n')
            lines = [ln.lstrip() for ln in s.split('\n')]

            # Ensure blank line before first table header (line that starts with '| ')
            header_idx = None
            for idx, ln in enumerate(lines):
                if ln.startswith('| '):
                    header_idx = idx
                    break
            if header_idx is not None and header_idx > 0 and lines[header_idx - 1].strip() != '':
                lines.insert(header_idx, '')
                header_idx += 1

            # Ensure separator row exists immediately after header and is valid
            if header_idx is not None:
                sep_idx = header_idx + 1
                if not (sep_idx < len(lines) and re.match(r"^\|\s*-+", lines[sep_idx])):
                    # build a simple separator with three dashes per column
                    cols = [c for c in lines[header_idx].split('|') if c.strip()]
                    sep = '|' + '|'.join(['---' for _ in cols]) + '|'
                    lines.insert(sep_idx, sep)

            # Re-join ensuring each table row is on its own line
            formatted_boq = '\n'.join(lines).strip() + '\n\n'
        except Exception:
            # If normalization fails, fall back to original
            formatted_boq = formatted_boq

        # (Previously attempted to insert formatted BOQ into vector_store; removed to avoid mutating index)

        logger.info("Comprehensive BOQ extraction completed successfully")
        return formatted_boq
        
    except Exception as e:
        logger.error(f"Error in comprehensive BOQ extraction: {e}")
        raise