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
    
    def __init__(self, api_key: str, model_name: str = None, temperature: float = None):
        """
        Initialize LLM client.
        
        Args:
            api_key: Google API key. Required.
            model_name: Model to use. Defaults to config value.
            temperature: Sampling temperature. Defaults to config value.
        """
        self.api_key = api_key
        self.model_name = model_name or settings.llm.model_name
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        
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
        try:
            result = str(self.llm.invoke(prompt))
        except Exception as e:
            if "API_KEY_INVALID" in str(e):
                raise ValueError("Invalid API key")
            raise
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
            results.append(self.invoke(prompt))
        return results
