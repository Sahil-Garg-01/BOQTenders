"""
Pydantic schemas for API request/response validation.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat endpoint request schema."""
    question: str = Field(..., description="Question to ask about the document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the total quantity of steel required?"
            }
        }


class ChatResponse(BaseModel):
    """Chat endpoint response schema."""
    answer: str = Field(..., description="Answer to the question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The total quantity of steel required is 500 MT."
            }
        }


class GetBoqResponse(BaseModel):
    """Get BOQ endpoint response schema."""
    message: str = Field(..., description="Status message")
    output: str = Field(..., description="Extracted BOQ in markdown format")
    consistency_score: float = Field(..., ge=0, le=100, description="Consistency score as percentage (0-100)")
    runs: int = Field(..., description="Number of runs performed")
    successful_runs: int = Field(..., description="Number of successful runs")
    avg_confidence: float = Field(..., ge=0, le=100, description="Average confidence score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "success",
                "output": "## DOCUMENT SUMMARY\n...",
                "consistency_score": 92.5,
                "runs": 2,
                "successful_runs": 2,
                "avg_confidence": 85.2
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "No file provided",
                "detail": "Please provide a PDF file"
            }
        }
