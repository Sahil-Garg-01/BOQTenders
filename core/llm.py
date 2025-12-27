"""
LLM client wrapper for Google Generative AI (Gemini).
"""
from typing import Optional
from loguru import logger
from langchain_google_genai import GoogleGenerativeAI

from config.settings import settings


class LLMClient:
    """
    Wrapper for LLM interactions with Google Generative AI.
    
    Example:
        client = LLMClient()
        response = client.invoke("What is a BOQ?")
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, api_key: str = None):
        """
        Initialize LLM client.
        
        Args:
            model_name: Model to use. Defaults to config value.
            temperature: Sampling temperature. Defaults to config value.
            api_key: Google API key. Defaults to config value.
        """
        self.model_name = model_name or settings.llm.model_name
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.api_key = api_key or settings.llm.google_api_key
        
        self._llm: Optional[GoogleGenerativeAI] = None
    
    @property
    def llm(self) -> GoogleGenerativeAI:
        """Lazy-load LLM instance."""
        if self._llm is None:
            logger.info(f'Initializing LLM: {self.model_name} (temp={self.temperature})')
            self._llm = GoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key
            )
            logger.info('LLM initialized successfully')
        return self._llm
    
    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and get a response.
        
        Args:
            prompt: The prompt text.
        
        Returns:
            LLM response as string.
        """
        logger.debug(f'Invoking LLM with prompt of length {len(prompt)}')
        result = str(self.llm.invoke(prompt))
        logger.debug(f'LLM response received, length: {len(result)}')
        return result
    
    def batch_invoke(self, prompts: list[str]) -> list[str]:
        """
        Send multiple prompts to the LLM.
        
        Args:
            prompts: List of prompt texts.
        
        Returns:
            List of LLM responses.
        """
        logger.info(f'Batch invoking LLM with {len(prompts)} prompts')
        results = []
        for i, prompt in enumerate(prompts, 1):
            logger.debug(f'Processing prompt {i}/{len(prompts)}')
            results.append(self.invoke(prompt))
        return results
