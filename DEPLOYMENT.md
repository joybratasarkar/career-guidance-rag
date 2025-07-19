# 🚀 Deployment Guide for Impacteers RAG System

This guide covers local development and production deployment of the Impacteers RAG system with Streamlit frontend.

## 📋 Prerequisites

- Docker and Docker Compose installed
- Google Cloud service account key (`xooper.json`)
- Environment variables configured

## 🔧 Environment Setup

### 1. Create `.env` file

```bash
# Google Cloud Configuration
GOOGLE_CREDENTIALS_PATH=./xooper.json
PROJECT_ID=xooper-450012
LOCATION=us-central1

# MongoDB Configuration (for local development)
MONGO_URI=mongodb://admin:password@mongodb:27017/impacteers_rag?authSource=admin

# Redis Configuration (for local development)
REDIS_URL=redis://redis:6379/0

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
API_TITLE=Impacteers RAG API
API_DESCRIPTION=RAG system for Impacteers career platform
API_VERSION=1.0.0
API_HOST=0.0.0.0
API_PORT=8000

# System Configuration
LOG_LEVEL=INFO
MAX_CONVERSATION_HISTORY=5
ENABLE_CORS=true
```

### 2. Place Google Cloud Credentials

```bash
# Place your Google Cloud service account key
cp path/to/your/credentials.json xooper.json
```

## 🐳 Docker Deployment

### Quick Start

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### Manual Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🌐 Access Points

After deployment, access your application at:

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MongoDB Express**: http://localhost:8081 (admin/admin)

## 📱 Free Deployment Options

### 1. Railway.app

1. **Connect GitHub**: Link your repository
2. **Environment Variables**: Add all .env variables
3. **Deploy**: Railway auto-deploys from main branch
4. **Custom Domain**: Get a free .railway.app subdomain

```bash
# Railway CLI deployment
npm install -g @railway/cli
railway login
railway init
railway up
```

### 2. Render.com

1. **Web Service**: Create new web service from GitHub
2. **Build Command**: `docker-compose build`
3. **Start Command**: `docker-compose up`
4. **Environment**: Add environment variables
5. **Free Tier**: 512MB RAM, sleeps after 15min inactivity

### 3. Fly.io

1. **Install Fly CLI**: Install flyctl
2. **Login**: `fly auth login`
3. **Initialize**: `fly launch`
4. **Deploy**: `fly deploy`

```bash
# Fly.io deployment
curl -L https://fly.io/install.sh | sh
fly auth login
fly launch --copy-config --name impacteers-rag
fly deploy
```

### 4. DigitalOcean App Platform

1. **Create App**: From GitHub repository
2. **Configure Services**: 
   - Backend: Dockerfile.backend
   - Frontend: Dockerfile.frontend
3. **Environment Variables**: Add all required vars
4. **Deploy**: Auto-deploy on commits

## 🔧 Production Configuration

### Environment Variables for Production

```bash
# Production MongoDB (MongoDB Atlas)
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/impacteers_rag

# Production Redis (Redis Cloud)
REDIS_URL=redis://username:password@host:port/0

# Production settings
LOG_LEVEL=WARNING
ENABLE_CORS=false
```

### Scaling Considerations

1. **Database**: Use MongoDB Atlas for production
2. **Redis**: Use Redis Cloud or AWS ElastiCache
3. **File Storage**: Use cloud storage for documents
4. **Load Balancing**: Use reverse proxy (nginx)
5. **SSL**: Enable HTTPS with Let's Encrypt

## 📊 Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:8501/_stcore/health

# Database health
docker-compose exec mongodb mongosh --eval "db.runCommand('ping')"

# Redis health
docker-compose exec redis redis-cli ping
```

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs -f frontend
docker-compose logs -f backend
docker-compose logs -f mongodb
docker-compose logs -f redis

# Follow logs in real-time
docker-compose logs -f --tail=100
```

## 🛠️ Development

### Local Development

```bash
# Start only dependencies
docker-compose up mongodb redis -d

# Run backend locally
python main.py

# Run frontend locally (in another terminal)
streamlit run streamlit_app.py

# Run CLI
python cli.py chat
```

### Testing

```bash
# Run tests
pytest

# Test API endpoints
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "session_id": "test"}'

# Test WebSocket
websocat ws://localhost:8000/ws/test_user
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Increase Docker memory limit
3. **Authentication errors**: Check Google credentials
4. **Connection errors**: Verify MongoDB/Redis URLs
5. **Build errors**: Check Python version compatibility

### Debug Commands

```bash
# Check container status
docker-compose ps

# Inspect container
docker-compose exec backend bash

# Check environment variables
docker-compose exec backend env

# Restart specific service
docker-compose restart backend

# Rebuild without cache
docker-compose build --no-cache backend
```

## 📈 Performance Optimization

1. **Redis Configuration**: Tune memory settings
2. **MongoDB Indexing**: Ensure proper indexes
3. **Connection Pooling**: Configure pool sizes
4. **Caching**: Implement response caching
5. **CDN**: Use CDN for static assets

## 🔒 Security

1. **Environment Variables**: Never commit secrets
2. **Authentication**: Add API authentication
3. **Rate Limiting**: Implement rate limiting
4. **HTTPS**: Use SSL in production
5. **Firewall**: Configure firewall rules

This deployment setup provides a complete, production-ready chat system with Redis conversation storage, MongoDB document storage, and a beautiful Streamlit frontend! 🎉