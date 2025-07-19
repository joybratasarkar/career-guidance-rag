#!/bin/bash

# Deployment script for Impacteers RAG System

echo "🚀 Starting Impacteers RAG System Deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please create it first."
    echo "📋 Required environment variables:"
    echo "   - MONGO_URI"
    echo "   - PROJECT_ID"
    echo "   - LOCATION" 
    echo "   - REDIS_URL"
    echo "   - GOOGLE_CREDENTIALS_PATH"
    exit 1
fi

# Check if Google credentials file exists
if [ ! -f xooper.json ]; then
    echo "❌ Google credentials file (xooper.json) not found!"
    echo "📋 Please place your Google Cloud service account key as 'xooper.json'"
    exit 1
fi

# Create necessary directories
mkdir -p logs

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start all services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check backend
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Backend (FastAPI) is healthy"
else
    echo "❌ Backend (FastAPI) is not responding"
fi

# Check frontend
if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✅ Frontend (Streamlit) is healthy"
else
    echo "❌ Frontend (Streamlit) is not responding"
fi

# Check MongoDB
if docker-compose exec mongodb mongosh --eval "db.runCommand('ping')" >/dev/null 2>&1; then
    echo "✅ MongoDB is healthy"
else
    echo "❌ MongoDB is not responding"
fi

# Check Redis
if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Show running containers
echo ""
echo "📊 Running containers:"
docker-compose ps

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📱 Access your application:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   MongoDB Express: http://localhost:8081 (admin/admin)"
echo ""
echo "🔍 Monitor logs:"
echo "   docker-compose logs -f frontend"
echo "   docker-compose logs -f backend"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"