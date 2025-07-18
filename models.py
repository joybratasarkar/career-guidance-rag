"""
Data models for the Impacteers RAG system
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Document types"""
    FAQ = "faq"
    FEATURE = "feature"
    POLICY = "policy"
    MANUAL = "manual"


class Category(str, Enum):
    """Document categories"""
    JOB_SEARCH = "job_search_opportunities"
    COURSES = "courses_upskilling"
    SKILL_ASSESSMENT = "skill_assessment"
    MENTORSHIP = "mentorship"
    COMMUNITY = "community_events"
    FEATURES = "platform_features"


class QueryType(str, Enum):
    """Query types"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    CONVERSATIONAL = "conversational"
    COMPLEX = "complex"


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "I'm looking for a job in tech",
                "session_id": "user_123_session"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Session ID")
    retrieved_docs: int = Field(..., description="Number of retrieved documents")
    context_used: bool = Field(..., description="Whether context was used")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "I'd be happy to help you find tech jobs! Before exploring job opportunities, please sign up to get personalized job suggestions.",
                "session_id": "user_123_session",
                "retrieved_docs": 3,
                "context_used": True,
                "processing_time": 1.23,
                "error": None
            }
        }


class DocumentInput(BaseModel):
    """Input model for document ingestion"""
    content: str = Field(..., description="Document content")
    document_type: DocumentType = Field(..., description="Type of document")
    category: Category = Field(..., description="Document category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Q: I'm looking for a job. A: Please sign up to get personalized job suggestions.",
                "document_type": "faq",
                "category": "job_search_opportunities",
                "metadata": {"source": "impacteers_faq", "priority": "high"}
            }
        }


class IngestionRequest(BaseModel):
    """Ingestion request model"""
    documents: List[DocumentInput] = Field(..., description="List of documents to ingest")
    
    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "Q: I'm looking for a job. A: Please sign up to get personalized job suggestions.",
                        "document_type": "faq",
                        "category": "job_search_opportunities",
                        "metadata": {"source": "impacteers_faq"}
                    }
                ]
            }
        }


class IngestionResponse(BaseModel):
    """Ingestion response model"""
    success: bool = Field(..., description="Whether ingestion was successful")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "documents_processed": 5,
                "chunks_created": 23,
                "processing_time": 15.67,
                "error": None
            }
        }


class EvaluationResponse(BaseModel):
    """Evaluation response model"""
    success: bool = Field(..., description="Whether evaluation was successful")
    overall_score: float = Field(..., description="Overall system score")
    retrieval_metrics: Dict[str, float] = Field(..., description="Retrieval performance metrics")
    generation_metrics: Dict[str, float] = Field(..., description="Generation quality metrics")
    test_cases_count: int = Field(..., description="Number of test cases evaluated")
    evaluation_report: str = Field(..., description="Detailed evaluation report")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "overall_score": 0.85,
                "retrieval_metrics": {
                    "avg_precision": 0.82,
                    "avg_recall": 0.78,
                    "avg_f1_score": 0.80
                },
                "generation_metrics": {
                    "avg_relevance": 0.88,
                    "avg_accuracy": 0.87,
                    "avg_helpfulness": 0.89
                },
                "test_cases_count": 6,
                "evaluation_report": "System performing well with good retrieval and generation quality...",
                "processing_time": 45.23,
                "error": None
            }
        }


class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: str = Field(..., description="System health status")
    documents_stored: int = Field(..., description="Number of documents in database")
    conversations_saved: int = Field(..., description="Number of conversations saved")
    last_check: datetime = Field(..., description="Last health check timestamp")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "documents_stored": 150,
                "conversations_saved": 45,
                "last_check": "2024-01-15T10:30:00Z",
                "error": None
            }
        }


class ConversationHistory(BaseModel):
    """Conversation history model"""
    session_id: str = Field(..., description="Session ID")
    user_query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    timestamp: datetime = Field(..., description="Conversation timestamp")
    retrieved_docs: int = Field(..., description="Number of retrieved documents")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "user_123_session",
                "user_query": "I'm looking for a job",
                "response": "I'd be happy to help you find jobs!",
                "timestamp": "2024-01-15T10:30:00Z",
                "retrieved_docs": 3
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Document not found",
                "error_type": "NotFoundError",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class Document(BaseModel):
    """Internal document model"""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: List[float] = Field(default_factory=list, description="Document embedding")
    category: str = Field(..., description="Document category")
    document_type: str = Field(..., description="Document type")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc_123",
                "content": "Q: I'm looking for a job. A: Please sign up...",
                "chunks": [{"content": "chunk1", "embedding": [0.1, 0.2]}],
                "metadata": {"source": "faq", "category": "job_search"},
                "embedding": [0.1, 0.2, 0.3],
                "category": "job_search_opportunities",
                "document_type": "faq",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }