#!/usr/bin/env python3
"""
Memory-optimized startup script for Render deployment
"""
import os
import gc
import sys
import uvicorn
from typing import Optional

# Memory optimization
os.environ.setdefault('TORCH_HOME', '/tmp/torch')
os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/transformers')
os.environ.setdefault('HF_HOME', '/tmp/huggingface')

# Lazy loading of heavy imports
_app = None
_embedding_service = None

def get_app():
    """Lazy load the FastAPI app to reduce startup memory"""
    global _app
    if _app is None:
        # Import only when needed
        from main import app
        _app = app
        # Force garbage collection
        gc.collect()
    return _app

def optimize_memory():
    """Apply memory optimizations"""
    import torch
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(1)  # Limit CPU threads
    
    # Force garbage collection
    gc.collect()

if __name__ == "__main__":
    # Apply optimizations
    optimize_memory()
    
    # Get the app
    app = get_app()
    
    # Start uvicorn with memory-conscious settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=1,  # Single worker to save memory
        loop="asyncio",
        access_log=False,  # Disable access logs to save memory
    )