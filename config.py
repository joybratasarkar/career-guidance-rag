"""
Configuration settings for the Impacteers RAG system
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer





class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings"""
    def get_embedding_model(
        model_name: str = "sentence-transformers/all-MiniLM-L6-V2"
    ) -> SentenceTransformer:
        """
        Return a SentenceTransformer embedding model.
        If you want to switch models, just change the name or add logic
        to pick a model based on environment variables.
        """
        return SentenceTransformer(model_name)
    # Google Cloud Configuration
    google_credentials_path: str = Field(
        default=os.path.abspath("xooper.json"),
        description="Path to Google Cloud service account key file"
    )
    project_id: str = Field(default="xooper-450012", description="Google Cloud Project ID")
    location: str = Field(default="us-central1", description="Google Cloud region")
    
    # MongoDB Configuration
    mongo_uri: str = Field(
        default="mongodb+srv://xooper:lsBAmSmNcI0s7uUW@xoopercluster.alvrs.mongodb.net/?retryWrites=true&w=majority&appName=xoopercluster",
        description="MongoDB connection URI"
    )
    database_name: str = Field(default="impacteers_rag", description="MongoDB database name")
    
    # Model Configuration
    llm_model: str = Field(default="gemini-2.0-flash-001", description="LLM model name")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-V2",
        description="Embedding model name"
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature")
    
    # RAG Configuration
    chunk_size: int = Field(default=800, description="Text chunk size")
    chunk_overlap: int = Field(default=100, description="Chunk overlap size")
    max_retrieval_docs: int = Field(default=5, description="Maximum documents to retrieve")
    max_context_length: int = Field(default=2000, description="Maximum context length")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold")
    
    # API Configuration
    api_title: str = Field(default="Impacteers RAG API", description="API title")
    api_description: str = Field(default="RAG system for Impacteers career platform", description="API description")
    api_version: str = Field(default="1.0.0", description="API version")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis server hostname")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_username: Optional[str] = Field(default=None, description="Redis username")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # System Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_conversation_history: int = Field(default=5, description="Maximum conversation history")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        if not os.path.exists(self.google_credentials_path):
            raise FileNotFoundError(
                f"Google credentials file not found: {self.google_credentials_path}"
            )
        
        if not self.mongo_uri:
            raise ValueError("MongoDB URI not configured")
        
        return True


# Global settings instance
settings = Settings()

# Set environment variable for Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials_path
