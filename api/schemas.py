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


class UploadResponse(BaseModel):
    """Upload endpoint response schema."""
    message: str = Field(..., description="Status message")
    output: str = Field(..., description="Extracted BOQ in markdown format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "success",
                "output": "## DOCUMENT SUMMARY\n..."
            }
        }


class ConsistencyResponse(BaseModel):
    """Consistency check endpoint response schema."""
    consistency_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Consistency score as percentage (0-100)"
    )
    successful_runs: int = Field(..., description="Number of successful runs")
    avg_confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Average confidence score"
    )
    is_low_consistency: bool = Field(
        ...,
        description="Whether consistency is below threshold"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "consistency_score": 92.5,
                "successful_runs": 4,
                "avg_confidence": 85.2,
                "is_low_consistency": False
            }
        }


class BOQResponse(BaseModel):
    """BOQ extraction response schema."""
    boq_output: str = Field(..., description="Extracted BOQ in markdown format")
    items_count: int = Field(default=0, description="Number of BOQ items extracted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "boq_output": "## DOCUMENT SUMMARY\n...",
                "items_count": 25
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
                "message": "No file uploaded",
                "detail": "Please upload a PDF file"
            }
        }
