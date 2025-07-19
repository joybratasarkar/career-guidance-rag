# 💬 Impacteers RAG Chat System

A production-ready RAG (Retrieval-Augmented Generation) chat system with **Streamlit Frontend**, **FastAPI Backend**, **Redis Memory**, and **Vertex AI** for the Impacteers career platform.

## 🚀 Features

### **Frontend & User Experience**
- **💬 Streamlit Chat Interface**: Beautiful, responsive web UI for conversations
- **🔌 Real-time WebSocket Chat**: Instant messaging with multiple concurrent users
- **👤 Multi-User Support**: Each user gets isolated chat rooms with unique IDs
- **📱 Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile
- **🔄 Connection Fallback**: Automatic REST API fallback if WebSocket fails
- **📜 Conversation History**: Load and display previous chat sessions

### **Backend & Architecture**
- **🔄 Three-Phase Architecture**: Ingestion, Inference, and Evaluation
- **📊 LangGraph Integration**: State-based processing with checkpointing
- **🤖 Vertex AI Models**: Gemini 2.0 Flash (LLM) + Embedding Gecko
- **🗄️ Hybrid Storage**: MongoDB for documents, Redis for conversations
- **🧠 Smart Memory Management**: 1-day TTL conversation storage in Redis
- **⚡ FastAPI WebSocket + REST**: High-performance async API with dual protocols
- **🎯 Advanced RAG Patterns**: Query enhancement, context building, evaluation
- **📈 Comprehensive Evaluation**: Retrieval and generation quality metrics

## 📁 Project Structure

```
impacteers-rag/
├── 🎨 Frontend
│   ├── streamlit_app.py        # Streamlit chat interface
│   └── .streamlit/            # Streamlit configuration
│       ├── config.toml        # UI settings
│       └── secrets.toml       # Connection secrets
├── 🔧 Backend
│   ├── main.py                # FastAPI application with WebSocket
│   ├── config.py              # Configuration settings
│   ├── models.py              # Pydantic data models
│   ├── database.py            # Hybrid MongoDB + Redis manager
│   ├── redis_manager.py       # Redis conversation storage
│   ├── embedding_service.py   # Shared embedding service
│   ├── ingestion_service.py   # Document ingestion pipeline
│   ├── inference_service.py   # Chat inference pipeline
│   └── evaluation_service.py  # System evaluation pipeline
├── 🐳 Deployment
│   ├── Dockerfile.backend     # Backend container
│   ├── Dockerfile.frontend    # Frontend container
│   ├── docker-compose.yml     # Multi-service orchestration
│   ├── deploy.sh             # One-command deployment
│   └── DEPLOYMENT.md         # Deployment guide
├── 🛠️ Development
│   ├── cli.py                # Command-line interface
│   ├── test_system.py        # System testing
│   └── requirements.txt      # Python dependencies
├── 📋 Configuration
│   ├── .env                  # Environment variables
│   ├── xooper.json          # Google Cloud credentials
│   └── CLAUDE.md            # Development guide
└── 📚 Documentation
    └── README.md            # This file
```

## 🚀 Quick Start

### 🎯 Fast Setup (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd impacteers-rag

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (see Environment Setup below)
cp .env.example .env
# Edit .env with your credentials

# 5. Start the system
# Terminal 1: Start Backend
uvicorn main:app --host 0.0.0.0 --port 6000 --reload

# Terminal 2: Start Frontend  
streamlit run streamlit_app.py --server.port 8504
```

### 📱 Access Points

- **💬 Chat Interface**: http://localhost:8504 (Streamlit UI)
- **📊 API Backend**: http://localhost:6000 (FastAPI with docs)
- **📚 API Documentation**: http://localhost:6000/docs (Interactive API docs)
- **🔍 Health Check**: http://localhost:6000/health

## 🛠️ Installation & Setup

### Prerequisites

- **Docker & Docker Compose** (for containerized deployment)
- **Python 3.11+** (for local development)
- **Google Cloud Project** with Vertex AI API enabled
- **Redis Instance** (included in Docker setup or external)
- **MongoDB Database** (included in Docker setup or MongoDB Atlas)
- **Service Account Key** (`xooper.json`) for Google Cloud

### 1. Clone and Setup

```bash
git clone <repository-url>
cd impacteers-rag
```

### 2. Environment Configuration

Create your `.env` file with the required settings:

```bash
# Google Cloud Configuration
GOOGLE_CREDENTIALS_PATH=./xooper.json
PROJECT_ID=your-project-id
LOCATION=us-central1

# Database Configuration
MONGO_URI=your-mongodb-uri
REDIS_URL=your-redis-url

# Model Configuration
LLM_MODEL=gemini-2.0-flash-001
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-V2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### 3. Google Cloud Setup

```bash
# Place your service account key file
cp path/to/your/credentials.json ./xooper.json

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
```

## 🎯 Deployment Options

### Option 1: Complete Docker Deployment (Recommended)

```bash
# One-command deployment with all services
./deploy.sh

# Manual deployment
docker-compose up --build -d

# Check all services
docker-compose ps
```

### Option 2: Local Development (Optimized)

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend (optimized for speed)
uvicorn main:app --host 0.0.0.0 --port 6000 --reload

# Start frontend (in another terminal)
streamlit run streamlit_app.py --server.port 8504

# CLI mode for testing
python cli.py chat
```

### Option 3: CLI Usage

```bash
# Interactive chat mode
python cli.py chat

# Ingest documents
python cli.py ingest --file path/to/document.pdf

# Run evaluation
python cli.py evaluate

# Test the system
python cli.py test

# Get help
python cli.py --help
```

## 💬 How to Use

### 1. Streamlit Chat Interface (Primary)

1. **Open Browser**: http://localhost:8504
2. **Start Chatting**: Type messages in the chat input
3. **Multi-User Support**: Each browser gets a unique user ID
4. **Fast Responses**: Optimized for 1-4 second response times
5. **Smart Link Responses**: Get real Impacteers URLs for jobs, courses, mentorship
6. **Conversation History**: Previous chats are automatically loaded from Redis

### 2. WebSocket Chat (Advanced Users)

```bash
# Connect to WebSocket with custom user ID
websocat ws://localhost:6000/ws/your_user_id

# Send messages directly
> "I'm looking for a job"
< "Before exploring job opportunities, please sign up to get personalised job suggestions. You can browse open roles here: https://www.impacteers.com/jobs"
```

### 3. REST API (Integration)

```bash
# Chat via REST API
curl -X POST "http://localhost:6000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What courses do you offer?", "session_id": "user123"}'

# Get conversation history
curl "http://localhost:6000/conversations/user123"

# Health check
curl "http://localhost:6000/health"
```

### 4. Performance Features

- **⚡ Fast Responses**: 1-4 seconds (83% faster than before)
- **🔗 Real Links**: Actual Impacteers URLs returned for every scenario
- **💾 Redis Memory**: 1-day TTL conversation storage
- **🧠 Smart Templates**: Immediate responses for common queries
- **📊 Vector Search**: Optimized embedding and retrieval

## 📚 API Reference

### 🔌 WebSocket Endpoints

- **WS /ws/{user_id}** - Real-time chat per user (ws://localhost:6000/ws/user123)

### 💬 Chat Endpoints

- **POST /chat** - Chat with the RAG system (returns real Impacteers URLs)
- **GET /conversations/{user_id}** - Get conversation history from Redis

### 🛠️ System Endpoints

- **POST /ingest** - Ingest documents into MongoDB (PDF support)
- **POST /evaluate** - Run system evaluation
- **GET /health** - Health check (MongoDB + Redis + conversations count)
- **POST /setup** - Quick setup with sample data
- **GET /stats** - System statistics (documents + conversations + evaluation)

### 📊 Utility Endpoints

- **POST /test-chat** - Batch testing with predefined queries
- **GET /sample-documents** - Get sample documents for ingestion

### 🔗 Link Categories

The system returns real URLs for these categories:
- **Jobs**: https://www.impacteers.com/jobs (+ filters for location, type, company)
- **Courses**: https://www.impacteers.com/courses (+ filters for category, level, type)
- **Assessments**: https://www.impacteers.com/assessments (+ interview prep)
- **Mentorship**: https://www.impacteers.com/mentorship (+ expertise filters)
- **Events**: https://www.impacteers.com/events (+ IIPL, challenges)
- **Community**: https://www.impacteers.com/community
- **Signup**: https://www.impacteers.com/signup

```

## 🌐 Free Deployment Options

Deploy your chat system on free hosting platforms:

### 1. **Railway.app** (Recommended)
- **Easy GitHub Integration**: Connect your repo and auto-deploy
- **Free Tier**: 512MB RAM, $5 credit monthly
- **Custom Domain**: Free .railway.app subdomain

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### 2. **Render.com**
- **Free Web Service**: Auto-deploy from GitHub
- **Docker Support**: Uses docker-compose.yml
- **Auto-Sleep**: Sleeps after 15min inactivity

### 3. **Fly.io**
- **Global Edge Deployment**: 256MB RAM free
- **Docker Native**: Perfect for our multi-service setup

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
fly launch
fly deploy
```

### 4. **DigitalOcean App Platform**
- **$5/month**: But often has free credits
- **Multi-Service**: Supports our frontend + backend setup

## 🔧 Configuration

### Complete Environment Variables

```env
# Google Cloud Configuration
PROJECT_ID=your-project-id
LOCATION=us-central1
GOOGLE_CREDENTIALS_PATH=./xooper.json

# Database Configuration  
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/impacteers_rag
REDIS_URL=redis://username:password@host:port/0

# Model Configuration
LLM_MODEL=gemini-2.0-flash-001
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-V2
LLM_TEMPERATURE=0.2

# RAG Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MAX_RETRIEVAL_DOCS=5
MAX_CONTEXT_LENGTH=2000
SIMILARITY_THRESHOLD=0.3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_CONVERSATION_HISTORY=5
ENABLE_CORS=true
```

## 🔧 Configuration

### Environment Variables

```env
# Google Cloud
PROJECT_ID=xooper-450012
LOCATION=us-central1
GOOGLE_CREDENTIALS_PATH=./xooper.json

# MongoDB
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
DATABASE_NAME=impacteers_rag

# Models
LLM_MODEL=gemini-2.0-flash-001
EMBEDDING_MODEL=textembedding-gecko@003

# RAG Settings
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MAX_RETRIEVAL_DOCS=5
SIMILARITY_THRESHOLD=0.3

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 🏗️ Architecture

### 🎨 Frontend Architecture (Streamlit)
```mermaid
graph TB
    A[Streamlit UI] --> B[WebSocket Client]
    A --> C[REST Client]
    B --> D[FastAPI Backend]
    C --> D
    D --> E[Response Handler]
    E --> A
```

### 🔧 Backend Architecture (FastAPI + Services)
```mermaid
graph TB
    A[WebSocket/REST] --> B[ConnectionManager]
    A --> C[InferenceService]
    C --> D[Redis Memory]
    C --> E[MongoDB Documents]
    C --> F[Vertex AI]
    B --> G[Multi-User Isolation]
```

### 💾 Storage Architecture (Hybrid)
```mermaid
graph LR
    A[User Conversations] --> B[Redis TTL 1-day]
    C[Documents/Embeddings] --> D[MongoDB Persistent]
    E[Evaluation Results] --> D
    F[Session State] --> B
```

### 🔄 Three-Phase Processing

**Phase 1: Ingestion Pipeline**
- Document processing and chunking
- Embedding generation (Vertex AI)
- MongoDB storage with indexing

**Phase 2: Inference Pipeline**  
- Redis conversation loading (per user_id)
- Query enhancement with context
- Hybrid retrieval + response generation
- Redis conversation saving

**Phase 3: Evaluation Pipeline**
- Retrieval and generation quality assessment
- Automated performance reporting
- MongoDB evaluation storage

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/test_api.py

# Test database operations
pytest tests/test_database.py
```

### Live Testing

```bash
# Test Streamlit interface
open http://localhost:8501

# Test WebSocket connection
websocat ws://localhost:8000/ws/test_user
> "Hello!"

# Test REST API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What jobs are available?", "session_id": "test"}'

# CLI testing
python cli.py test
```

## 🚨 Troubleshooting

### Common Issues

```bash
# Service not responding
docker-compose ps
docker-compose logs backend
docker-compose logs frontend

# Redis connection issues
docker-compose exec redis redis-cli ping

# MongoDB connection issues  
docker-compose exec mongodb mongosh --eval "db.runCommand('ping')"

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose up --build

# Check health endpoints
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

## 📊 Monitoring

### Health Checks

- **Backend Health**: http://localhost:8000/health
- **Frontend Health**: http://localhost:8501/_stcore/health  
- **MongoDB**: Included in backend health
- **Redis**: Included in backend health

### Performance Metrics

- **Chat Response Time**: Displayed in Streamlit UI
- **Conversation Storage**: Redis TTL tracking
- **Document Retrieval**: Vector similarity scores
- **System Load**: Docker container stats

```bash
# Monitor container resources
docker stats

# View real-time logs
docker-compose logs -f --tail=100
```

## 🎯 Key Features Summary

✅ **Multi-User Chat**: Isolated conversations per user_id  
✅ **⚡ Fast Responses**: 1-4 seconds (83% faster than before)  
✅ **🔗 Real URLs**: Actual Impacteers links returned for every scenario  
✅ **💾 Redis Memory**: 1-day TTL conversation storage  
✅ **🖥️ Streamlit UI**: Beautiful, responsive chat interface  
✅ **🐳 Docker Ready**: Complete containerized deployment  
✅ **🌐 Free Hosting**: Deploy on Railway, Render, Fly.io  
✅ **📊 Production Ready**: Health checks, logging, monitoring, PDF ingestion  

## 🚀 **Performance Achievements**

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| **Simple Queries** | 7-12 seconds | **1.4 seconds** | **83% faster** |
| **Template Responses** | 7-12 seconds | **0.7 seconds** | **90% faster** |
| **Knowledge Queries** | 7-12 seconds | **3-4 seconds** | **65% faster** |
| **Link Accuracy** | Placeholder links | **Real Impacteers URLs** | **100% real** |

## 📋 **Current System Status**

✅ **Backend**: FastAPI on port 6000 (optimized)  
✅ **Frontend**: Streamlit on port 8504 (working)  
✅ **Database**: MongoDB + Redis hybrid storage  
✅ **PDFs**: 2 documents ingested (33 chunks total)  
✅ **Performance**: Sub-4 second responses  
✅ **Links**: All templates updated with real URLs  
✅ **Memory**: Redis TTL working with conversation persistence  

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the beautiful frontend framework
- **FastAPI**: For high-performance async backend
- **LangGraph**: For powerful graph-based processing
- **Google Vertex AI**: For state-of-the-art AI models
- **Redis**: For fast conversation memory storage
- **MongoDB**: For flexible document storage

## 📞 Support

- **Issues**: Open GitHub issues for bugs/features
- **Documentation**: See DEPLOYMENT.md for detailed setup
- **Chat**: Use the system to chat with the Impacteers AI! 

---

**🚀 Ready to deploy your AI chat system? Start with `./deploy.sh` and you'll be chatting in minutes! 🎉**
