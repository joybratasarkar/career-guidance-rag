"""
FastAPI application for the Impacteers RAG system (with WebSocket chat support)
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
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
ingestion_service: IngestionService = None  # Initialized in lifespan
evaluation_service: EvaluationService = None
inference_service: InferenceService = None

# WebSocket connection manager per user
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(user_id, []).append(websocket)
        logger.info(f"WebSocket connected for user: {user_id}")

    def disconnect(self, user_id: str, websocket: WebSocket):
        connections = self.active_connections.get(user_id, [])
        if websocket in connections:
            connections.remove(websocket)
            logger.info(f"WebSocket disconnected for user: {user_id}")
        if not connections:
            self.active_connections.pop(user_id, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            # Find and remove the failed connection
            for user_id, connections in self.active_connections.items():
                if websocket in connections:
                    self.disconnect(user_id, websocket)
                    break

    async def broadcast_to_user(self, user_id: str, message: str):
        connections = self.active_connections.get(user_id, []).copy()  # Copy to avoid modification during iteration
        failed_connections = []
        
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                failed_connections.append(connection)
        
        # Remove failed connections
        for failed_connection in failed_connections:
            self.disconnect(user_id, failed_connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting up Impacteers RAG system...")
    settings.validate_config()
    await init_database()

    global ingestion_service, inference_service, evaluation_service
    db_manager: DatabaseManager = await get_database()
    ingestion_service = IngestionService(db_manager)
    inference_service = InferenceService(db_manager)
    evaluation_service = EvaluationService(db_manager, inference_service)

    logger.info("RAG system started successfully")
    yield

    logger.info("Shutting down RAG system...")
    await close_database()
    logger.info("RAG system shut down complete")

# Instantiate FastAPI
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# CORS
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=str(exc), error_type=type(exc).__name__).dict()
    )

# Health check
@app.get("/health", response_model=SystemStatusResponse)
async def health_check(db: DatabaseManager = Depends(get_database)):
    try:
        health = await db.health_check()
        return SystemStatusResponse(
            status=health["status"],
            documents_stored=health["documents_count"],
            conversations_saved=health["conversations_count"],
            last_check=health["last_check"],
            error=health.get("error")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Handle WebSocket connections per user"""
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Create chat request and get response with user_id
            req = ChatRequest(query=data, session_id=user_id)
            res = await inference_service.chat(req, user_id=user_id)  # Pass user_id explicitly
            # Send back via WebSocket
            await manager.send_personal_message(res.response, websocket)
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)

# REST chat fallback
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    # For REST endpoint, use session_id as user_id for compatibility
    return await inference_service.chat(request, user_id=request.session_id)

# Ingest documents
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest, background_tasks: BackgroundTasks):
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    if len(request.documents) > 10:
        background_tasks.add_task(_background_ingestion, request.documents)
        return IngestionResponse(
            success=True, documents_processed=len(request.documents),
            chunks_created=0, processing_time=0.0, error="Background processing"
        )
    result = await ingestion_service.ingest_documents(request.documents)
    return IngestionResponse(
        success=result["success"], documents_processed=result["documents_processed"],
        chunks_created=result["chunks_created"], processing_time=0.0, error=result.get("error")
    )

async def _background_ingestion(documents: List[DocumentInput]):
    try:
        out = await ingestion_service.ingest_documents(documents)
        logger.info(f"Background ingestion done: {out}")
    except Exception as e:
        logger.error(f"Background ingestion error: {e}")

# Evaluate system
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system():
    if not evaluation_service:
        raise HTTPException(status_code=503, detail="Evaluation service not available")
    return await evaluation_service.run_evaluation()

# Conversation history
@app.get("/conversations/{user_id}", response_model=List[ConversationHistory])
async def get_conversation_history(user_id: str):
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    hist = await inference_service.get_conversation_history(user_id)
    return [ConversationHistory(
        session_id=user_id,  # Use user_id as session_id for compatibility
        user_query=conv["user_query"],
        response=conv["response"], 
        timestamp=conv["timestamp"],
        retrieved_docs=len(conv.get("retrieved_docs", []))  # Count docs directly
    ) for conv in hist]

# Sample documents
@app.get("/sample-documents", response_model=List[DocumentInput])
async def get_sample_documents():
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    return await ingestion_service.get_sample_documents()

# Quick setup
@app.post("/setup")
async def quick_setup() -> Dict[str, Any]:
    if not ingestion_service or not evaluation_service:
        raise HTTPException(status_code=503, detail="Services not available")
    samples = await ingestion_service.get_sample_documents()
    ing = await ingestion_service.ingest_documents(samples)
    req = ChatRequest(query="I'm looking for a job")
    inf = await inference_service.chat(req)
    eval_res = await evaluation_service.run_evaluation()
    return {
        "setup_complete": True,
        "ingestion": ing,
        "inference_test": {"query": req.query, "response": inf.response, "retrieved_docs": inf.retrieved_docs},
        "evaluation": {"success": eval_res.success, "overall_score": eval_res.overall_score, "test_cases": eval_res.test_cases_count}
    }

# System stats
@app.get("/stats")
async def get_system_stats(db: DatabaseManager = Depends(get_database)) -> Dict[str, Any]:
    try:
        docs = await db.get_documents_count()
        convs = await db.get_conversations_count()
        latest = await db.get_latest_evaluation()
        return {
            "documents_count": docs,
            "conversations_count": convs,
            "latest_evaluation": {
                "score": latest.get("overall_metrics", {}).get("system_score", 0.0) if latest else 0.0,
                "timestamp": latest.get("created_at") if latest else None
            }
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch test
@app.post("/test-chat", response_model=List[ChatResponse])
async def test_chat_batch():
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    queries = [
        "I'm looking for a job", "What courses do you offer?", "How assess skills?",
        "Tell me about mentorship", "What's IIPL?", "How improve my resume?"
    ]
    responses = []
    sid = "test_batch_session"
    for q in queries:
        resp = await inference_service.chat(ChatRequest(query=q, session_id=sid))
        responses.append(resp)
    return responses

# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
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
            "test": "/test-chat",
            "websocket": "/ws/{user_id}"
        }
    }

# Run app
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
