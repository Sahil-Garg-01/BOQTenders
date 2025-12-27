"""
PDF text extraction module using external API.
"""
import os
from typing import Optional
import requests
from loguru import logger

from config.settings import settings


class PDFExtractor:
    """
    Handles PDF text extraction using HuggingFace Space API.
    
    Example:
        extractor = PDFExtractor()
        text = extractor.extract_text("document.pdf")
    """
    
    def __init__(self, api_url: str = None, api_token: str = None, timeout: int = None):
        """
        Initialize PDF extractor.
        
        Args:
            api_url: URL for extraction API. Defaults to config value.
            api_token: HuggingFace API token. Defaults to config value.
            timeout: Request timeout in seconds. Defaults to config value.
        """
        self.api_url = api_url or settings.pdf.extraction_api_url
        self.api_token = api_token or settings.pdf.hf_api_token
        self.timeout = timeout or settings.pdf.request_timeout
    
    def extract_text(self, pdf_path: str, start_page: int = None, end_page: int = None, filename: str = None) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            start_page: Starting page number (1-indexed). Defaults to config value.
            end_page: Ending page number. Defaults to config value.
            filename: Display name for logging. Defaults to basename of pdf_path.
        
        Returns:
            Extracted text content.
        
        Raises:
            requests.RequestException: If API request fails.
            FileNotFoundError: If PDF file doesn't exist.
        """
        # Use config defaults if not specified
        start_page = start_page or settings.pdf.start_page
        end_page = end_page or settings.pdf.end_page
        display_name = filename or os.path.basename(pdf_path)
        
        logger.info(f'Starting text extraction for {display_name} (pages {start_page}-{end_page})')
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {
                'start_page': start_page,
                'end_page': end_page,
                'filename': os.path.basename(pdf_path)
            }
            headers = {'Authorization': f'Bearer {self.api_token}'}
            
            response = requests.post(
                self.api_url,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout
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
    
    def extract_text_preview(self, pdf_path: str, max_chars: int = 200) -> str:
        """
        Extract and return a preview of the PDF text.
        
        Args:
            pdf_path: Path to the PDF file.
            max_chars: Maximum characters to return.
        
        Returns:
            Preview of extracted text.
        """
        text = self.extract_text(pdf_path, start_page=1, end_page=5)
        return text[:max_chars] + "..." if len(text) > max_chars else text
