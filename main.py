"""
FastAPI application for the Impacteers RAG system
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from models import (
    ChatRequest, ChatResponse, IngestionRequest, IngestionResponse,
    EvaluationResponse, SystemStatusResponse, ConversationHistory,
    ErrorResponse, DocumentInput
)
from database import DatabaseManager, get_database, init_database, close_database
from ingestion_service import IngestionService
from inference_service import InferenceService
from evaluation_service import EvaluationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
ingestion_service = None
inference_service = None
evaluation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    # Startup
    logger.info("Starting up Impacteers RAG system...")
    
    # Validate configuration
    settings.validate_config()
    
    # Initialize database
    await init_database()
    
    # Initialize services
    global ingestion_service, inference_service, evaluation_service
    db_manager = await get_database()
    
    ingestion_service = IngestionService(db_manager)
    inference_service = InferenceService(db_manager)
    evaluation_service = EvaluationService(db_manager, inference_service)
    
    logger.info("RAG system started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG system...")
    await close_database()
    logger.info("RAG system shut down complete")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=SystemStatusResponse)
async def health_check(db: DatabaseManager = Depends(get_database)):
    """Health check endpoint"""
    try:
        health_status = await db.health_check()
        
        return SystemStatusResponse(
            status=health_status["status"],
            documents_stored=health_status["documents_count"],
            conversations_saved=health_status["conversations_count"],
            last_check=health_status["last_check"],
            error=health_status.get("error")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system"""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")
        
        response = await inference_service.chat(request)
        return response
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion endpoint
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Ingest documents into the RAG system"""
    try:
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Ingestion service not available")
        
        # Run ingestion in background for large datasets
        if len(request.documents) > 10:
            background_tasks.add_task(
                _background_ingestion, 
                request.documents
            )
            return IngestionResponse(
                success=True,
                documents_processed=len(request.documents),
                chunks_created=0,
                processing_time=0.0,
                error="Processing in background"
            )
        
        # Process immediately for small datasets
        result = await ingestion_service.ingest_documents(request.documents)
        
        return IngestionResponse(
            success=result["success"],
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            processing_time=0.0,  # Will be calculated in service
            error=result.get("error")
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _background_ingestion(documents: List[DocumentInput]):
    """Background task for ingestion"""
    try:
        result = await ingestion_service.ingest_documents(documents)
        logger.info(f"Background ingestion completed: {result}")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")


# Evaluation endpoint
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system():
    """Evaluate the RAG system performance"""
    try:
        if not evaluation_service:
            raise HTTPException(status_code=503, detail="Evaluation service not available")
        
        response = await evaluation_service.run_evaluation()
        return response
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation history endpoint
@app.get("/conversations/{session_id}", response_model=List[ConversationHistory])
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")
        
        history = await inference_service.get_conversation_history(session_id)
        
        return [
            ConversationHistory(
                session_id=conv["session_id"],
                user_query=conv["user_query"],
                response=conv["response"],
                timestamp=conv["timestamp"],
                retrieved_docs=conv.get("retrieved_docs_count", 0)
            )
            for conv in history
        ]
    
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sample documents endpoint
@app.get("/sample-documents", response_model=List[DocumentInput])
async def get_sample_documents():
    """Get sample documents for testing"""
    try:
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Ingestion service not available")
        
        return await ingestion_service.get_sample_documents()
    
    except Exception as e:
        logger.error(f"Failed to get sample documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quick setup endpoint
@app.post("/setup", response_model=Dict[str, Any])
async def quick_setup():
    """Quick setup with sample data"""
    try:
        if not ingestion_service or not evaluation_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get sample documents
        sample_docs = await ingestion_service.get_sample_documents()
        
        # Ingest sample documents
        ingestion_result = await ingestion_service.ingest_documents(sample_docs)
        
        # Test inference
        test_request = ChatRequest(query="I'm looking for a job")
        inference_result = await inference_service.chat(test_request)
        
        # Run evaluation
        evaluation_result = await evaluation_service.run_evaluation()
        
        return {
            "setup_complete": True,
            "ingestion": {
                "success": ingestion_result["success"],
                "documents_processed": ingestion_result["documents_processed"],
                "chunks_created": ingestion_result["chunks_created"]
            },
            "inference_test": {
                "query": test_request.query,
                "response": inference_result.response,
                "retrieved_docs": inference_result.retrieved_docs
            },
            "evaluation": {
                "success": evaluation_result.success,
                "overall_score": evaluation_result.overall_score,
                "test_cases": evaluation_result.test_cases_count
            }
        }
    
    except Exception as e:
        logger.error(f"Quick setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System statistics endpoint
@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats(db: DatabaseManager = Depends(get_database)):
    """Get system statistics"""
    try:
        docs_count = await db.get_documents_count()
        convs_count = await db.get_conversations_count()
        latest_eval = await db.get_latest_evaluation()
        
        return {
            "documents_count": docs_count,
            "conversations_count": convs_count,
            "latest_evaluation": {
                "score": latest_eval.get("overall_metrics", {}).get("system_score", 0.0) if latest_eval else 0.0,
                "timestamp": latest_eval.get("created_at") if latest_eval else None
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Interactive testing endpoint
@app.post("/test-chat", response_model=List[ChatResponse])
async def test_chat_batch():
    """Test chat with multiple queries"""
    try:
        if not inference_service:
            raise HTTPException(status_code=503, detail="Inference service not available")
        
        test_queries = [
            "I'm looking for a job",
            "What courses do you offer?",
            "How can I assess my skills?",
            "Tell me about mentorship",
            "What's IIPL?",
            "How do I improve my resume?"
        ]
        
        responses = []
        session_id = "test_batch_session"
        
        for query in test_queries:
            request = ChatRequest(query=query, session_id=session_id)
            response = await inference_service.chat(request)
            responses.append(response)
        
        return responses
    
    except Exception as e:
        logger.error(f"Batch test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Impacteers RAG System API",
        "version": settings.api_version,
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "chat": "/chat",
            "ingest": "/ingest",
            "evaluate": "/evaluate",
            "setup": "/setup",
            "stats": "/stats",
            "test": "/test-chat"
        }
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )