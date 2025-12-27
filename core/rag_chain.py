"""
RAG chain builder for conversational retrieval.
"""
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from core.llm import LLMClient
from prompts.get_prompts import QA_TEMPLATE


class RAGChainBuilder:
    """
    Builder for RAG (Retrieval-Augmented Generation) chains.
    
    Example:
        builder = RAGChainBuilder()
        chain = builder.build(vector_store)
        response = chain({"question": "What is the total quantity?"})
    """
    
    def __init__(self, llm_client: LLMClient = None):
        """
        Initialize RAG chain builder.
        
        Args:
            llm_client: LLM client instance. Creates new one if not provided.
        """
        self.llm_client = llm_client or LLMClient()
    
    def build(self, vector_store: FAISS, qa_template: str = None, memory_key: str = "chat_history", return_messages: bool = True) -> ConversationalRetrievalChain:
        """
        Build a conversational retrieval chain.
        
        Args:
            vector_store: FAISS vector store with document embeddings.
            qa_template: Custom Q&A prompt template. Defaults to standard template.
            memory_key: Key for conversation memory.
            return_messages: Whether to return messages in memory.
        
        Returns:
            Configured ConversationalRetrievalChain.
        """
        logger.info('Building RAG chain with LangChain classic API')
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages
        )
        
        # Create prompt
        template = qa_template or QA_TEMPLATE
        qa_prompt = PromptTemplate.from_template(template)
        
        # Build chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_client.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': qa_prompt},
        )
        
        logger.info('RAG chain built successfully')
        return chain
    
    def build_simple_retriever(self, vector_store: FAISS, k: int = 4):
        """
        Build a simple retriever without conversation memory.
        
        Args:
            vector_store: FAISS vector store.
            k: Number of documents to retrieve.
        
        Returns:
            Retriever instance.
        """
        return vector_store.as_retriever(search_kwargs={"k": k})
