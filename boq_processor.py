import os
from typing import List, Optional
import dotenv
from pydantic_settings import BaseSettings
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import importlib
import time
from functools import wraps
import re
import requests

from langchain_core.documents import Document

# Load environment variables
dotenv.load_dotenv()

def retry_with_exponential_backoff(max_retries: int = 3, initial_delay: int = 2, backoff_factor: int = 2):
    """Decorator for retry logic with exponential backoff for API rate limits."""
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
                    is_rate_limit = ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower())
                    if is_rate_limit and attempt < max_retries:
                        delay = initial_delay * (backoff_factor ** attempt)
                        logger.warning(f"Rate limit encountered. Retry {attempt + 1}/{max_retries} in {delay}s: {error_str}")
                        time.sleep(delay)
                    else:
                        if is_rate_limit and attempt == max_retries:
                            logger.error(f"Rate limit exhausted after {max_retries} retries")
                        raise
            raise last_exception
        return wrapper
    return decorator

class Settings(BaseSettings):
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    hf_api_token: str = os.getenv("HF_API_TOKEN")
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 500
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()

class BOQProcessor:
    def __init__(self):
        self.settings = settings
        self.llm = GoogleGenerativeAI(model=self.settings.model_name, temperature=self.settings.temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        self.lc_old_api: Optional[bool] = None

    def _table_to_markdown(self, table: dict) -> str:
        # Commented out: Table extraction not needed for now
        """
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        if not headers:
            return ''
        md = '| ' + ' | '.join(headers) + ' |\n'
        md += '|' + '|'.join(['---'] * len(headers)) + '|\n'
        for row in rows:
            md += '| ' + ' | '.join(str(cell) for cell in row) + ' |\n'
        return md
        """

    def _call_extract_text_api(self, pdf_path: str, start_page: int = 1, end_page: int = 100, filename: str = None) -> str:
        display_name = filename or os.path.basename(pdf_path)
        logger.info(f'Starting text extraction for {display_name} (pages {start_page}-{end_page})')
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {'start_page': start_page, 'end_page': end_page, 'filename': os.path.basename(pdf_path)}
            headers = {'Authorization': f'Bearer {self.settings.hf_api_token}'}
            response = requests.post(
                'https://point9-extract-text-and-table.hf.space/api/text',
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            json_response = response.json()
            if isinstance(json_response, dict):
                result = json_response.get('result', '')
            else:
                logger.error(f"Unexpected response format: {json_response}")
                result = ''
            logger.info(f'Text extraction completed, response length: {len(result)}')
            return result

    def _call_extract_tables_api(self, pdf_path: str, start_page: int = 1, end_page: int = 2, filename: str = None) -> List[dict]:
        # Commented out: Table extraction not needed for now
        """
        display_name = filename or os.path.basename(pdf_path)
        logger.info(f'Starting table extraction for {display_name} (pages {start_page}-{end_page})')
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {'start_page': start_page, 'end_page': end_page, 'filename': os.path.basename(pdf_path)}
            headers = {'Authorization': f'Bearer {self.settings.hf_api_token}'}
            response = requests.post(
                'https://point9-extract-text-and-table.hf.space/api/tables',
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            json_response = response.json()
            if isinstance(json_response, dict):
                result = json_response.get('result', [])
                # Filter to only include valid table dicts
                valid_tables = [t for t in result if isinstance(t, dict)]
                invalid_count = len(result) - len(valid_tables)
                if invalid_count > 0:
                    logger.warning(f"Filtered out {invalid_count} invalid tables (not dicts)")
                result = valid_tables
            else:
                logger.error(f"Unexpected response format: {json_response}")
                result = []
            logger.info(f'Table extraction completed, found {len(result)} valid tables')
            return result
        """

    def load_and_process_pdf(self, pdf_path: str, filename: str = None) -> List[Document]:
        try:
            display_name = filename or os.path.basename(pdf_path)
            logger.info(f'Processing PDF from {display_name} using Hugging Face API')
            logger.info('Calling text extraction API...')
            extracted_text = self._call_extract_text_api(pdf_path, filename=filename)
            logger.info(f'Extracted text length: {len(extracted_text)}')
            if extracted_text:
                logger.info(f'Text preview: {extracted_text[:200]}...')
            else:
                logger.warning('Extracted text is empty')
            # Commented out: Table extraction not needed for now
            """
            logger.info('Calling table extraction API...')
            tables = self._call_extract_tables_api(pdf_path, filename=filename)
            logger.info(f'Extracted {len(tables)} tables')
            logger.info('Converting tables to markdown...')
            table_texts = [self._table_to_markdown(table) for table in tables]
            logger.info(f'Converted {len(table_texts)} tables to markdown')
            full_content = extracted_text + '\n\n' + '\n\n'.join(table_texts)
            """
            full_content = extracted_text
            logger.info(f'Combined content length: {len(full_content)}')
            logger.info('Splitting content into chunks...')
            chunks = self.text_splitter.create_documents([full_content])
            logger.info(f'Split into {len(chunks)} chunks')
            return chunks
        except Exception as e:
            logger.error(f'Error loading and processing PDF: {e}')
            raise

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        try:
            logger.info('Creating embeddings and vector store')
            logger.info(f'Processing {len(chunks)} chunks for embeddings')
            embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
            logger.info('Embeddings model loaded, creating FAISS vector store...')
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info('Vector store created successfully')
            return vector_store
        except Exception as e:
            logger.error(f'Error creating vector store: {e}')
            raise

    def _detect_langchain_api(self):
        logger.info('Detecting LangChain API version...')
        try:
            conv_module = importlib.import_module('langchain_classic.chains')
            if hasattr(conv_module, 'ConversationalRetrievalChain'):
                self.lc_old_api = True
                logger.info('Detected LangChain classic API (0.1.x)')
                from langchain_classic.chains import ConversationalRetrievalChain
                from langchain_classic.memory import ConversationBufferMemory
                from langchain_core.prompts import PromptTemplate
                return ConversationalRetrievalChain, ConversationBufferMemory, PromptTemplate
        except ImportError:
            pass
        try:
            create_retrieval_chain = importlib.import_module('langchain.chains').create_retrieval_chain
            create_history_aware_retriever = importlib.import_module('langchain.chains').create_history_aware_retriever
            create_stuff_documents_chain = importlib.import_module('langchain.chains').create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            self.lc_old_api = False
            logger.info('Detected LangChain new API (1.x)')
            return create_retrieval_chain, create_history_aware_retriever, create_stuff_documents_chain, ChatPromptTemplate, MessagesPlaceholder
        except ImportError as e:
            logger.error('Unsupported LangChain version')
            raise RuntimeError('Unsupported LangChain version. Install 0.1.x or 1.x series.') from e

    def setup_rag_chain(self, vector_store: FAISS):
        logger.info('Setting up RAG chain')
        api_components = self._detect_langchain_api()
        if self.lc_old_api:
            logger.info('Using old LangChain API for RAG chain setup')
            ConversationalRetrievalChain, ConversationBufferMemory, PromptTemplate = api_components
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            qa_template = '''You are an expert assistant specializing in construction and tender documents, with deep knowledge of Bill of Quantities (BOQ) analysis. Your role is to provide accurate, helpful, and professional responses based solely on the provided context.

Guidelines:
- Always base your answers on the given context. Do not use external knowledge or assumptions.
- For BOQ-related questions, provide detailed, structured information including item codes, descriptions, quantities, units, rates, and amounts where available.
- If the context lacks specific information, respond with: "The requested information is not available in the provided document context."
- Be concise yet comprehensive. Structure responses clearly (e.g., use bullet points or tables for lists).
- Handle follow-up questions by referencing previous context in the conversation history.
- Maintain neutrality and professionalism in all responses.

{context}

Question: {question}
Answer:'''
            qa_prompt = PromptTemplate.from_template(qa_template)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={'prompt': qa_prompt},
            )
        else:
            logger.info('Using new LangChain API for RAG chain setup')
            create_retrieval_chain, create_history_aware_retriever, create_stuff_documents_chain, ChatPromptTemplate, MessagesPlaceholder = api_components
            contextualize_q_system_prompt = (
                'Given a chat history and the latest user question which might reference context in the chat history, '
                'formulate a standalone question which can be understood without the chat history. Do not answer the question, '
                'just reformulate it if needed and otherwise return it as is.'
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ('system', contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ])
            history_aware_retriever = create_history_aware_retriever(
                self.llm, vector_store.as_retriever(), contextualize_q_prompt
            )
            qa_system_prompt = '''You are an expert assistant specializing in construction and tender documents, with deep knowledge of Bill of Quantities (BOQ) analysis. Your role is to provide accurate, helpful, and professional responses based solely on the provided context.

Guidelines:
- Always base your answers on the given context. Do not use external knowledge or assumptions.
- For BOQ-related questions, provide detailed, structured information including item codes, descriptions, quantities, units, rates, and amounts where available.
- If the context lacks specific information, respond with: "The requested information is not available in the provided document context."
- Be concise yet comprehensive. Structure responses clearly (e.g., use bullet points or tables for lists).
- Handle follow-up questions by referencing previous context in the conversation history.
- Maintain neutrality and professionalism in all responses.

{context}'''
            qa_prompt = ChatPromptTemplate.from_messages([
                ('system', qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ])
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info(f'RAG chain set up successfully ({"old" if self.lc_old_api else "new"} API)')
        return qa_chain

    def _extract_metadata(self, chunks: List[Document]) -> str:
        metadata_text = '\n\n'.join([chunk.page_content for chunk in chunks[:3]])
        metadata_prompt = f'''Extract key information from this tender document excerpt in a concise format:
- Document Type
- Project Name
- Issuing Authority
- Tender Number
- Date
- Location

Document excerpt:
{metadata_text[:2000]}

Output only the facts, no extra analysis.'''
        logger.info('Invoking LLM for metadata extraction...')
        result = str(self.llm.invoke(metadata_prompt))
        logger.info('Metadata extraction completed')
        return result

    def _batch_chunks(self, chunks: List[Document], batch_size: int = 24) -> List[List[Document]]:
        return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    def _extract_boq_from_batch(self, batch_text: str, batch_chunks: List[Document], batch_num: int) -> List[str]:
        extraction_prompt = '''Analyze this text and extract ONLY Bill of Quantities (BOQ) line items if present.

Look for structured data with:
- Item numbers or codes
- Descriptions of work/materials
- Quantities
- Units (Nos, Sqm, Cum, m, etc.)
- Rates/Unit prices
- Total amounts

If you find BOQ items, return them in this EXACT format (pipe-separated):
ITEM_CODE|DESCRIPTION|QUANTITY|UNIT|RATE|AMOUNT|CONFIDENCE

Where CONFIDENCE is a score (0-100%) based on how clearly and completely the data appears in the text. Use lower scores (e.g., 70-90%) if information is partially missing, inferred, or unclear. Use 100% only for complete, directly stated data.

Rules for columns:
- If an entire column has no values, omit that column.
- For missing values, use "NA".

Return multiple items on separate lines. If NO BOQ items are found, return: "NO_BOQ_ITEMS"

Text to analyze:
{batch_text}

Extract only actual BOQ line items.'''

        prompt_text = batch_text[:30000]
        prompt = extraction_prompt.format(batch_text=prompt_text)

        try:
            logger.info(f'Invoking LLM for BOQ extraction on batch {batch_num}...')
            result = self.llm.invoke(prompt)
            logger.info(f'LLM response received for batch {batch_num}')
            if 'NO_BOQ_ITEMS' in str(result):
                logger.info(f'No BOQ items found in batch {batch_num}')
                return []
            boq_items = []
            lines = str(result).strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 7:
                    parts += ['NA'] * (7 - len(parts))
                source_tag = f'p.{batch_num}'
                if parts[1] and '(' not in parts[1]:
                    parts[1] = f'{parts[1]} ({source_tag})'
                boq_items.append('|'.join(parts[:7]))
            logger.info(f'Extracted {len(boq_items)} BOQ items from batch {batch_num}')
            return boq_items
        except Exception as e:
            logger.warning(f'Error processing batch {batch_num}: {e}')
            return []

    def _format_boq_output(self, unique_items: List[str], metadata_result: str) -> str:
        logger.info('Formatting BOQ output...')
        if not unique_items:
            logger.info('No BOQ items to format')
            return f'''## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
No BOQ items were found in this document.'''

        col_headers = ['Item No/Code', 'Description', 'Quantity', 'Unit', 'Rate', 'Amount', 'Confidence Score']
        cols_present = [False] * 7
        normalized_items = []
        for item in unique_items:
            parts = [p.strip() for p in item.split('|')]
            if len(parts) < 7:
                parts += ['NA'] * (7 - len(parts))
            normalized_items.append(parts[:7])
            for i in range(7):
                if parts[i] and parts[i].upper() != 'NA':
                    cols_present[i] = True

        col_indices = [i for i, present in enumerate(cols_present) if present]
        if 0 not in col_indices:
            col_indices.insert(0, 0)
        if 1 not in col_indices:
            col_indices.insert(1, 1)

        header_row = '| ' + ' | '.join([col_headers[i] for i in col_indices]) + ' |\n'
        sep_row = '|' + '|'.join(['-' * (len(col_headers[i]) + 2) for i in col_indices]) + '|\n'

        formatted_boq = f'''## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
**Total Items Found:** {len(unique_items)}

{header_row}{sep_row}'''

        for parts in normalized_items:
            parts[1] = parts[1][:80] if len(parts[1]) > 80 else parts[1]
            row_vals = [parts[i] for i in col_indices]
            # Add % to confidence score if present
            if 6 in col_indices:
                conf_idx = col_indices.index(6)
                if row_vals[conf_idx] != 'NA':
                    row_vals[conf_idx] += '%'
            formatted_boq += '| ' + ' | '.join(row_vals) + ' |\n'

        formatted_boq += f'\n## SUMMARY\n- **Total Items:** {len(unique_items)}\n'

        try:
            s = formatted_boq.replace('\r\n', '\n').replace('\r', '\n')
            lines = [ln.lstrip() for ln in s.split('\n')]
            header_idx = next((i for i, ln in enumerate(lines) if ln.startswith('| ')), None)
            if header_idx and header_idx > 0 and lines[header_idx - 1].strip():
                lines.insert(header_idx, '')
            if header_idx:
                sep_idx = header_idx + 1
                if not (sep_idx < len(lines) and re.match(r'^\|\s*-+', lines[sep_idx])):
                    cols = [c for c in lines[header_idx].split('|') if c.strip()]
                    sep = '|' + '|'.join(['---' for _ in cols]) + '|'
                    lines.insert(sep_idx, sep)
            formatted_boq = '\n'.join(lines).strip() + '\n\n'
        except Exception:
            pass

        return formatted_boq

    @retry_with_exponential_backoff(max_retries=3, initial_delay=2)
    def extract_boq_comprehensive(self, chunks: List[Document], vector_store: FAISS = None) -> str:
        try:
            logger.info(f'Starting comprehensive BOQ extraction from {len(chunks)} chunks')
            logger.info('Extracting document metadata...')
            metadata_result = self._extract_metadata(chunks)
            logger.info('Metadata extracted, creating batches...')
            batches = self._batch_chunks(chunks)
            logger.info(f'Created {len(batches)} batches')
            boq_items = []
            for batch_num, batch_chunks in enumerate(batches, 1):
                logger.info(f'Processing batch {batch_num}/{len(batches)} ({len(batch_chunks)} chunks)')
                chunk_texts = [chunk.page_content for chunk in batch_chunks]
                batch_text = '\n\n'.join(chunk_texts)
                logger.info(f'Batch text length: {len(batch_text)}')
                batch_items = self._extract_boq_from_batch(batch_text, batch_chunks, batch_num)
                boq_items.extend(batch_items)
                logger.info(f'Batch {batch_num} yielded {len(batch_items)} items')
            unique_items = list(dict.fromkeys(boq_items))
            logger.info(f'Found {len(unique_items)} unique BOQ items after deduplication')
            logger.info('Formatting BOQ output...')
            formatted_boq = self._format_boq_output(unique_items, metadata_result)
            logger.info('Comprehensive BOQ extraction completed successfully')
            return formatted_boq
        except Exception as e:
            logger.error(f'Error in comprehensive BOQ extraction: {e}')
            raise

def check_consistency(chunks: List[Document], vector_store: FAISS, runs: int = 4) -> dict:
    """Run extraction multiple times and compute variance."""
    from difflib import SequenceMatcher
    
    results = []
    for _ in range(runs):
        try:
            boq = extract_boq_comprehensive(chunks, vector_store)
            results.append(boq)
        except Exception as e:
            logger.warning(f"Consistency run failed: {e}")
            results.append("")
    
    # Variance: Average similarity between pairs
    similarities = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if results[i] and results[j]:
                sim = SequenceMatcher(None, results[i], results[j]).ratio()
                similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    consistency_score = avg_similarity * 100
    
    # Average confidence from per-item scores
    all_confidences = []
    for boq in results:
        if boq:
            lines = boq.split('\n')
            confidence_idx = None
            for line in lines:
                line = line.strip()
                if '|' in line and 'Confidence' in line and not line.startswith('| ---'):
                    # Header row: find index of Confidence
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if 'Confidence' in parts:
                        confidence_idx = parts.index('Confidence')
                        break
            if confidence_idx is not None:
                for line in lines:
                    if '|' in line and not line.startswith('| ---') and 'Confidence' not in line:
                        parts = [p.strip() for p in line.split('|')[1:-1]]
                        if len(parts) > confidence_idx:
                            try:
                                conf_str = parts[confidence_idx]
                                if conf_str and conf_str != 'NA':
                                    # Remove % if present
                                    conf_str = conf_str.rstrip('%')
                                    conf = float(conf_str)
                                    all_confidences.append(conf)
                            except (ValueError, IndexError):
                                pass
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    return {
        "consistency_score": round(consistency_score, 2),
        "runs": runs,
        "successful_runs": len([r for r in results if r]),
        "avg_similarity": round(avg_similarity, 2),
        "avg_confidence": round(avg_confidence, 2),  # New metric
        "total_confidence_scores": len(all_confidences)
    }

# Global instance for backward compatibility
processor = BOQProcessor()

# Backward compatibility functions
def load_and_process_pdf(pdf_path: str, filename: str = None) -> List[Document]:
    return processor.load_and_process_pdf(pdf_path, filename)

def create_vector_store(chunks: List[Document]) -> FAISS:
    return processor.create_vector_store(chunks)

def setup_rag_chain(vector_store: FAISS):
    return processor.setup_rag_chain(vector_store)

@retry_with_exponential_backoff(max_retries=3, initial_delay=2)
def extract_boq_comprehensive(chunks: List[Document], vector_store: FAISS = None) -> str:
    return processor.extract_boq_comprehensive(chunks, vector_store)