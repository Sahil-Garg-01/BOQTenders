"""
LLM client wrapper for Google Generative AI (Gemini).
"""
from typing import Optional
from contextlib import nullcontext
from loguru import logger
from langchain_google_genai import GoogleGenerativeAI

from config.settings import settings

# OpenTelemetry
try:
    from opentelemetry import trace
    _tracer = trace.get_tracer(__name__)
except ImportError:
    _tracer = None


class LLMClient:
    """
    Wrapper for LLM interactions with Google Generative AI.
    
    Example:
        client = LLMClient()
        response = client.invoke("What is a BOQ?")
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = None, temperature: float = None):
        """
        Initialize LLM client.
        
        Args:
            api_key: Google API key. If None, uses from settings.
            model_name: Model to use. Defaults to config value.
            temperature: Sampling temperature. Defaults to config value.
        """
        self.api_key = api_key or settings.GOOGLE_API_KEY
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
        with (_tracer.start_as_current_span("llm.invoke") if _tracer else nullcontext()) as span:
            if span:
                span.set_attribute("llm.model", self.model_name)
                span.set_attribute("llm.temperature", float(self.temperature))
                span.set_attribute("prompt.length", len(prompt or ""))
            try:
                result = str(self.llm.invoke(prompt))
            except Exception as e:
                if span:
                    span.record_exception(e)
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
        with (_tracer.start_as_current_span("llm.batch_invoke") if _tracer else nullcontext()) as span:
            if span:
                span.set_attribute("prompts.count", len(prompts))
            logger.info(f'Batch invoking LLM with {len(prompts)} prompts')
            results = []
            for i, prompt in enumerate(prompts, 1):
                results.append(self.invoke(prompt))
            return results
