# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Development
- **Start complete system**: `./deploy.sh` (recommended - all services)
- **Start backend only**: `python main.py` or `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- **Start frontend only**: `streamlit run streamlit_app.py --server.port 8501`
- **CLI interface**: `python cli.py <command>` (full-demo, setup, chat, test, evaluate, status)
- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `pytest` or `pytest --cov=. --cov-report=html`
- **System testing**: `python test_system.py`

### Docker Multi-Service Deployment
- **Complete deployment**: `./deploy.sh` (includes frontend, backend, MongoDB, Redis)
- **Manual deployment**: `docker-compose up --build -d`
- **View logs**: `docker-compose logs -f backend` or `docker-compose logs -f frontend`
- **Check status**: `docker-compose ps`
- **Individual service**: `docker-compose up backend` or `docker-compose up frontend`

### Access Points
- **Streamlit Chat UI**: http://localhost:8501 (primary interface)
- **FastAPI Backend**: http://localhost:8000 (API endpoints)
- **API Documentation**: http://localhost:8000/docs
- **MongoDB UI**: http://localhost:8081 (admin/admin)

### Quick Setup
- **Complete setup**: `./deploy.sh`
- **API setup only**: `curl -X POST "http://localhost:8000/setup"`
- **Frontend setup**: Open http://localhost:8501 and start chatting

## Architecture Overview

This is an Impacteers RAG chat system with Streamlit frontend, FastAPI backend, Redis memory, and Google Vertex AI.

### Frontend + Backend Architecture
- **Streamlit Frontend** (`streamlit_app.py`): Beautiful chat UI with WebSocket and REST support
- **FastAPI Backend** (`main.py`): WebSocket + REST API endpoints with multi-user support
- **Redis Memory** (`redis_manager.py`): 1-day TTL conversation storage per user_id
- **MongoDB Storage** (`database.py`): Document and evaluation persistent storage

### Three-Phase Processing
1. **Ingestion Pipeline** (`ingestion_service.py`): Document processing, chunking, embedding, MongoDB storage
2. **Inference Pipeline** (`inference_service.py`): Redis conversation loading, query processing, retrieval, response generation, Redis saving
3. **Evaluation Pipeline** (`evaluation_service.py`): Retrieval and generation quality assessment

### Multi-User Chat System
- **WebSocket Rooms**: Each user_id gets isolated chat room (`/ws/{user_id}`)
- **Conversation Isolation**: Redis keys `conversations:{user_id}` with 1-day TTL
- **Real-time Communication**: WebSocket for instant messaging, REST fallback
- **Session Management**: Persistent conversations across disconnections

### Key Technologies
- **Streamlit**: Frontend chat interface with real-time updates
- **WebSocket**: Real-time bidirectional communication
- **Redis**: Fast conversation memory with automatic expiration
- **LangGraph**: State-based processing with checkpointing
- **Google Vertex AI**: Gemini 2.0 Flash (LLM) + Embedding models
- **MongoDB**: Document vector storage with hybrid search
- **FastAPI**: Async WebSocket + REST API

## Configuration

### Environment Setup
1. Place Google Cloud service account key: `xooper.json`
2. Set `GOOGLE_APPLICATION_CREDENTIALS="./xooper.json"`
3. Configure MongoDB URI in `config.py` or `.env`

### Key Settings (`config.py`)
- **Models**: `gemini-2.0-flash-001` (LLM), `sentence-transformers/all-MiniLM-L6-V2` (embeddings)
- **RAG**: `chunk_size=800`, `chunk_overlap=100`, `max_retrieval_docs=5`
- **API**: Default port 8000, CORS enabled

### Required Environment Variables
- `PROJECT_ID`: Google Cloud project (default: xooper-450012)
- `MONGO_URI`: MongoDB connection string
- `LOCATION`: Google Cloud region (default: us-central1)

## Service Architecture

### Global Service Initialization
Services are initialized in `main.py` lifespan context:
```python
ingestion_service: IngestionService = None  # Line 34
inference_service: InferenceService = None  # Line 36
evaluation_service: EvaluationService = None  # Line 35
```

### LangGraph State Management
Each service uses LangGraph with state machines:
- **IngestionState**: `documents`, `processed_chunks`, `stored_count`, `error`, `stage`
- **InferenceState**: `query`, `retrieved_docs`, `response`, `conversation_history`
- **EvaluationState**: `test_cases`, `retrieval_results`, `generation_results`, `overall_metrics`

### Memory and Persistence
- **Conversation History**: Stored in MongoDB with session-based retrieval
- **Document Storage**: MongoDB with vector embeddings and metadata
- **LangGraph Checkpointing**: MemorySaver for state persistence across steps

## Data Models (`models.py`)

### Core Models
- **ChatRequest/ChatResponse**: Chat interactions with session management
- **DocumentInput**: Document ingestion with type/category classification
- **IngestionRequest/Response**: Batch document processing
- **EvaluationResponse**: Comprehensive evaluation metrics

### Document Types and Categories
- **Types**: FAQ, FEATURE, POLICY, MANUAL
- **Categories**: JOB_SEARCH, COURSES, SKILL_ASSESSMENT, MENTORSHIP, COMMUNITY, FEATURES

## API Endpoints

### Core Endpoints
- `POST /chat`: Chat with RAG system
- `POST /ingest`: Ingest documents
- `POST /evaluate`: Run system evaluation
- `GET /health`: Health check with database status

### WebSocket
- `WS /ws/{user_id}`: Real-time chat per user

### Utility Endpoints
- `POST /setup`: Quick setup with sample data
- `GET /conversations/{session_id}`: Conversation history
- `POST /test-chat`: Batch testing queries
- `GET /stats`: System statistics

## Development Notes

### Service Dependencies
- All services require initialized `DatabaseManager`
- `EvaluationService` depends on `InferenceService`
- Services use `SharedEmbeddingService` for model management

### Error Handling
- Global exception handler in FastAPI
- Service-level error handling with state tracking
- Comprehensive logging with timestamps

### Testing
- Unit tests with pytest and pytest-asyncio
- Integration tests via `test_system.py`
- CLI testing commands
- API endpoint testing with httpx

### Performance Considerations
- Async/await throughout for concurrent operations
- Background task processing for large document batches (>10 docs)
- Connection pooling with Motor MongoDB client
- Embedding model caching via SharedEmbeddingService