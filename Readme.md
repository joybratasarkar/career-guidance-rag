# Impacteers RAG System

A production-ready RAG (Retrieval-Augmented Generation) system built with **LangGraph**, **FastAPI**, and **Vertex AI** for the Impacteers career platform.

## 🚀 Features

- **🔄 Three-Phase Architecture**: Ingestion, Inference, and Evaluation
- **📊 LangGraph Integration**: State-based processing with checkpointing
- **🤖 Vertex AI Models**: Gemini 2.0 Flash (LLM) + Embedding Gecko
- **🗄️ MongoDB Vector Store**: Scalable document storage and retrieval
- **🧠 Short-term Memory**: Conversation history with persistent storage
- **⚡ FastAPI REST API**: High-performance async API endpoints
- **🎯 Advanced RAG Patterns**: Query enhancement, context building, evaluation
- **📈 Comprehensive Evaluation**: Retrieval and generation quality metrics

## 📁 Project Structure

```
impacteers-rag/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── models.py              # Pydantic data models
├── database.py            # MongoDB operations
├── ingestion_service.py   # Document ingestion pipeline
├── inference_service.py   # Chat inference pipeline
├── evaluation_service.py  # System evaluation pipeline
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Google Cloud Project with Vertex AI enabled
- MongoDB Atlas cluster or local MongoDB
- Service account key file (`xooper.json`)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd impacteers-rag
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Google Cloud Setup

```bash
# Place your service account key file
cp path/to/your/xooper.json ./xooper.json

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="./xooper.json"

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
```

## 🚀 Quick Start

### Option 1: Using CLI (Recommended)

```bash
# Full demo with setup, testing, and evaluation
python cli.py full-demo

# Or run individual commands
python cli.py setup      # Setup system with sample data
python cli.py chat       # Interactive chat mode
python cli.py test       # Batch testing
python cli.py evaluate   # Run evaluation
python cli.py status     # System status
```

### Option 2: Using FastAPI

```bash
# Start the API server
python main.py

# Or using uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Option 3: Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f impacteers-rag
```

## 📚 API Endpoints

### Core Endpoints

- **POST /chat** - Chat with the RAG system
- **POST /ingest** - Ingest documents
- **POST /evaluate** - Run system evaluation
- **GET /health** - Health check

### Utility Endpoints

- **GET /conversations/{session_id}** - Get conversation history
- **POST /setup** - Quick setup with sample data
- **POST /test-chat** - Batch testing
- **GET /stats** - System statistics
- **GET /sample-documents** - Get sample documents

### Example Usage

```bash
# Chat with the system
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "I am looking for a job", "session_id": "user123"}'

# Quick setup
curl -X POST "http://localhost:8000/setup"

# Health check
curl "http://localhost:8000/health"
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

### Phase 1: Ingestion Pipeline

```mermaid
graph LR
    A[Documents] --> B[Process]
    B --> C[Chunk]
    C --> D[Embed]
    D --> E[Store]
```

- **Document Processing**: Converts FAQ and feature docs into structured format
- **Smart Chunking**: Creates overlapping chunks for better retrieval
- **Embedding Generation**: Uses Vertex AI Embedding Gecko
- **MongoDB Storage**: Stores with proper indexing

### Phase 2: Inference Pipeline

```mermaid
graph LR
    A[User Query] --> B[Load History]
    B --> C[Process Query]
    C --> D[Retrieve Docs]
    D --> E[Build Context]
    E --> F[Generate Response]
    F --> G[Save Conversation]
```

- **Query Processing**: Enhances queries with conversation context
- **Hybrid Retrieval**: Combines vector and text search
- **Context Building**: Intelligently compresses retrieved documents
- **Response Generation**: Uses Gemini 2.0 Flash for natural responses
- **Memory Management**: Maintains conversation history

### Phase 3: Evaluation Pipeline

```mermaid
graph LR
    A[Test Cases] --> B[Evaluate Retrieval]
    B --> C[Evaluate Generation]
    C --> D[Calculate Metrics]
    D --> E[Generate Report]
```

- **Retrieval Evaluation**: Measures precision, recall, F1-score
- **Generation Quality**: Assesses relevance, accuracy, helpfulness
- **Comprehensive Metrics**: Overall system performance scoring
- **Automated Reports**: LLM-generated insights and recommendations

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

### Manual Testing

```bash
# CLI testing
python cli.py test

# API testing
curl -X POST "http://localhost:8000/test-chat"
```

## 📊 Monitoring and Evaluation

### Metrics

- **Retrieval Metrics**: Precision, Recall, F1-Score, Similarity
- **Generation Metrics**: Relevance, Accuracy, Helpfulness
- **System Metrics**: Overall Score, Response Time, Error Rate

### Evaluation Dashboard

```bash
# Run evaluation
python cli.py evaluate

# View results
curl "http://localhost:8000/stats"
```

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🔒 Security

- **Authentication**: Add API key authentication
- **Rate Limiting**: Implement request rate limiting
- **Data Validation**: Pydantic models for input validation
- **Error Handling**: Comprehensive error handling and logging

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t impacteers-rag .

# Run container
docker run -p 8000:8000 impacteers-rag
```

### Cloud Deployment

```bash
# Google Cloud Run
gcloud run deploy impacteers-rag \
  --image gcr.io/PROJECT_ID/impacteers-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# AWS ECS, Azure Container Instances, etc.
```

### Production Considerations

- **Scaling**: Use load balancers and horizontal scaling
- **Monitoring**: Implement comprehensive monitoring
- **Logging**: Structured logging with aggregation
- **Backup**: Regular database backups
- **Security**: HTTPS, authentication, rate limiting

## 🛠️ Development

### Adding New Features

1. **New Model**: Add to `models.py`
2. **New Endpoint**: Add to `main.py`
3. **New Service**: Create new service file
4. **Tests**: Add corresponding tests

### Code Style

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

## 📈 Performance

### Optimization Tips

- **Batch Processing**: Process multiple documents together
- **Caching**: Cache frequently accessed data
- **Indexing**: Proper database indexing
- **Async Operations**: Use async/await throughout

### Benchmarks

- **Ingestion**: ~50 documents/second
- **Inference**: ~2-3 responses/second
- **Evaluation**: ~6 test cases in 45 seconds

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Check credentials file
   ls -la xooper.json
   export GOOGLE_APPLICATION_CREDENTIALS="./xooper.json"
   ```

2. **MongoDB Connection**
   ```bash
   # Test connection
   python -c "from pymongo import MongoClient; print(MongoClient('your-uri').admin.command('ping'))"
   ```

3. **Model Access**
   ```bash
   # Check Vertex AI access
   gcloud auth list
   gcloud services list --enabled | grep aiplatform
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size
   export CHUNK_SIZE=400
   export MAX_RETRIEVAL_DOCS=3
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the powerful graph-based processing framework
- **Google Cloud Vertex AI**: For state-of-the-art LLM and embedding models
- **MongoDB**: For flexible document storage and vector search capabilities
- **FastAPI**: For the high-performance async web framework
- **Impacteers Team**: For the domain expertise and use case requirements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@impacteers.com
- **Documentation**: [Full Documentation](https://docs.impacteers.com)

## 🔗 Links

- **Impacteers Platform**: [https://impacteers.com](https://impacteers.com)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **LangGraph Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Vertex AI Documentation**: [https://cloud.google.com/vertex-ai/docs](https://cloud.google.com/vertex-ai/docs)

## 🚀 Quick Demo

Get started in 3 steps:

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run demo
python cli.py full-demo
```

That's it! Your RAG system is ready for Impacteers! 🎉

---

**Built with ❤️ for the Impacteers career platform**