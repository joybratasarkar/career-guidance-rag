# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Development
- **Start API server**: `python main.py` or `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- **CLI interface**: `python cli.py <command>` (full-demo, setup, chat, test, evaluate, status)
- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `pytest` or `pytest --cov=. --cov-report=html`
- **System testing**: `python test_system.py`

### Docker
- **Build and run**: `docker-compose up -d`
- **View logs**: `docker-compose logs -f impacteers-rag`
- **Check status**: `docker-compose ps`

### Quick Setup
- **Full demo with setup**: `python cli.py full-demo`
- **API quick setup**: `curl -X POST "http://localhost:8000/setup"`

## Architecture Overview

This is an Impacteers RAG (Retrieval-Augmented Generation) system built with FastAPI, LangGraph, and Google Vertex AI.

### Three-Phase Architecture
1. **Ingestion Pipeline** (`ingestion_service.py`): Document processing, chunking, embedding, and storage
2. **Inference Pipeline** (`inference_service.py`): Query processing, retrieval, context building, and response generation
3. **Evaluation Pipeline** (`evaluation_service.py`): Retrieval and generation quality assessment

### Core Services
- **DatabaseManager** (`database.py`): MongoDB operations with async Motor client
- **SharedEmbeddingService** (`embedding_service.py`): Centralized embedding model management
- **FastAPI Application** (`main.py`): REST API and WebSocket endpoints

### Key Technologies
- **LangGraph**: State-based processing with checkpointing
- **Google Vertex AI**: Gemini 2.0 Flash (LLM) + Embedding Gecko models
- **MongoDB**: Vector storage with cosine similarity search
- **FastAPI**: Async REST API with WebSocket support
- **Sentence Transformers**: Local embedding fallback

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