"""
Configuration settings for the Impacteers RAG system
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") # Note: os.getenv returns strings, convert if needed



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
        default=MONGO_URI or "mongodb+srv://user:password@host/database",
        description="MongoDB connection URI"
    )
    database_name: str = Field(default="impacteers_rag", description="MongoDB database name")
    
    # Model Configuration
    llm_model: str = Field(default="gemini-2.0-flash-001", description="LLM model name")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-V2",
        description="Embedding model name"
    )
    llm_temperature: float = Field(default=0.1, description="LLM temperature (lower for faster, more deterministic responses)")
    llm_max_tokens: int = Field(default=256, description="Max tokens for faster responses")
    llm_timeout: int = Field(default=10, description="LLM request timeout in seconds")
    
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
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL", env="REDIS_URL")
    redis_host: str = Field(default="localhost", description="Redis server hostname", env="REDIS_HOST")
    redis_port: int = Field(default=6379, description="Redis server port", env="REDIS_PORT")
    redis_username: Optional[str] = Field(default=None, description="Redis username", env="REDIS_USERNAME")
    redis_password: Optional[str] = Field(default=None, description="Redis password", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, description="Redis database number", env="REDIS_DB")
    
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
        
        if not self.redis_host:
            raise ValueError("Redis host not configured")
        
        if not isinstance(self.redis_port, int) or not (1 <= self.redis_port <= 65535):
            raise ValueError(f"Invalid Redis port: {self.redis_port}")
        
        return True


# Global settings instance
settings = Settings()

# Set environment variable for Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials_path
