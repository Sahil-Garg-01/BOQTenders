"""
Embedding and vector store management module.
"""
from typing import List, Optional
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings


class EmbeddingService:
    """
    Handles text chunking, embeddings, and vector store operations.
    
    Example:
        service = EmbeddingService()
        chunks = service.split_text(text)
        vector_store = service.create_vector_store(chunks)
    """
    
    def __init__(self, embedding_model: str = None, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize embedding service.
        
        Args:
            embedding_model: HuggingFace model name. Defaults to config value.
            chunk_size: Size of text chunks. Defaults to config value.
            chunk_overlap: Overlap between chunks. Defaults to config value.
        """
        self.embedding_model = embedding_model or settings.embedding.embedding_model
        self.chunk_size = chunk_size or settings.embedding.chunk_size
        self.chunk_overlap = chunk_overlap or settings.embedding.chunk_overlap
        
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load embeddings model."""
        if self._embeddings is None:
            logger.info(f'Loading embeddings model: {self.embedding_model}')
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            logger.info('Embeddings model loaded successfully')
        return self._embeddings
    
    def split_text(self, text: str) -> List[Document]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Full text content to split.
        
        Returns:
            List of Document chunks.
        """
        logger.info(f'Splitting text of length {len(text)} into chunks...')
        chunks = self._text_splitter.create_documents([text])
        logger.info(f'Split into {len(chunks)} chunks')
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from document chunks.
        
        Args:
            chunks: List of Document objects.
        
        Returns:
            FAISS vector store instance.
        
        Raises:
            Exception: If vector store creation fails.
        """
        try:
            logger.info(f'Creating vector store from {len(chunks)} chunks')
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info('Vector store created successfully')
            return vector_store
        except Exception as e:
            logger.error(f'Error creating vector store: {e}')
            raise
    
    def add_documents(self, vector_store: FAISS, documents: List[Document]) -> None:
        """
        Add new documents to an existing vector store.
        
        Args:
            vector_store: Existing FAISS vector store.
            documents: Documents to add.
        """
        logger.info(f'Adding {len(documents)} documents to vector store')
        vector_store.add_documents(documents)
        logger.info('Documents added successfully')
    
    def similarity_search(
        self,
        vector_store: FAISS,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search on vector store.
        
        Args:
            vector_store: FAISS vector store to search.
            query: Search query.
            k: Number of results to return.
        
        Returns:
            List of most similar documents.
        """
        return vector_store.similarity_search(query, k=k)
