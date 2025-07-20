# 🚀 Render.com Deployment Guide

Deploy your Impacteers RAG chatbot to Render.com for **FREE** with this simple guide.

## 📋 Prerequisites

- GitHub repository: `joybratasarkar/career-guidance-rag`
- Google Cloud service account key (`xooper.json`)
- Render.com account (free)

## 🎯 Quick Deployment Steps

### Step 1: Go to Render.com
1. Visit https://render.com
2. Sign up/login with your GitHub account
3. Connect your GitHub repository

### Step 2: Deploy Backend (FastAPI)
1. Click **"New"** → **"Web Service"**
2. Connect repository: `joybratasarkar/career-guidance-rag`
3. Configure:
   - **Name**: `impacteers-backend`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile.backend`
   - **Plan**: `Free`
   - **Region**: `Oregon (US West)`

### Step 3: Add Environment Variables (Backend)
Add these in the "Environment" section:

```bash
PROJECT_ID=xooper-450012
LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./xooper.json
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MONGO_URI=mongodb://localhost:27017/impacteers_rag
REDIS_URL=redis://localhost:6379/0
```

### Step 4: Deploy Frontend (Streamlit)
1. Click **"New"** → **"Web Service"**
2. Connect same repository: `joybratasarkar/career-guidance-rag`
3. Configure:
   - **Name**: `impacteers-frontend`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile.frontend`
   - **Plan**: `Free`
   - **Region**: `Oregon (US West)`

### Step 5: Add Environment Variables (Frontend)
```bash
BACKEND_URL=https://impacteers-backend.onrender.com
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Step 6: Deploy Database Services (Optional)
For persistent storage, add:

1. **MongoDB**: Use MongoDB Atlas (free tier)
2. **Redis**: Use Redis Cloud (free tier)

## 🔧 Alternative: Simplified Single-Service Deployment

For easier deployment, use the `streamlit_free.py` version:

1. **Single Web Service**:
   - Main file: `streamlit_free.py`
   - No backend needed
   - **100% FREE** with Streamlit Cloud

2. **Deploy Command**:
   ```bash
   streamlit run streamlit_free.py --server.port $PORT --server.address 0.0.0.0
   ```

## 🌐 Access Your Deployed App

After deployment (5-10 minutes):

- **Frontend**: `https://impacteers-frontend.onrender.com`
- **Backend API**: `https://impacteers-backend.onrender.com`
- **API Docs**: `https://impacteers-backend.onrender.com/docs`

## 📊 Free Tier Limits

- **RAM**: 512MB per service
- **CPU**: Shared CPU
- **Sleep**: Services sleep after 15min inactivity
- **Build Time**: 500 hours/month
- **Bandwidth**: 100GB/month

## 🔧 Troubleshooting

### Common Issues:
1. **Build Failures**: Check Dockerfile paths
2. **Memory Issues**: Use lightweight model
3. **Connection Errors**: Verify environment variables
4. **Sleep Mode**: Services restart on first request

### Debug Commands:
```bash
# Check build logs in Render dashboard
# Monitor service health via Render UI
# Use simplified streamlit_free.py for testing
```

## 🎉 Success!

Your chatbot will be live at:
- **Chat Interface**: `https://your-app.onrender.com`
- **Complete REST API**: Available for integrations
- **Auto-deploy**: Updates on every git push

Perfect for demos, testing, and production use! 🚀