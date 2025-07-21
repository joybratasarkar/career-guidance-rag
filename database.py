"""
Database operations for the Impacteers RAG system
MongoDB for documents/evaluations, Redis for conversations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import IndexModel, TEXT
except ImportError as e:
    logging.error(f"MongoDB dependencies missing: {e}")
    raise ImportError("Please install: pip install motor pymongo") from e

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    logging.error(f"ML dependencies missing: {e}")
    raise ImportError("Please install: pip install scikit-learn numpy") from e

from config import settings
from models import Document
from redis_manager import RedisManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Hybrid database manager: MongoDB for documents/evaluations, Redis for conversations"""

    def __init__(self):
        # MongoDB for documents and evaluations
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.documents_collection = None
        self.evaluations_collection = None
        self._mongo_connected = False
        
        # Redis for conversations
        self.redis_manager = RedisManager()
        
        # Overall connection status
        self._is_connected = False

    async def connect(self):
        """Connect to both MongoDB and Redis"""
        if self._is_connected:
            return  # Already connected

        try:
            # Connect to MongoDB with proper error handling
            if not settings.mongo_uri or "mongodb" not in settings.mongo_uri.lower():
                raise ValueError(f"Invalid MongoDB URI: {settings.mongo_uri}")
                
            self.client = AsyncIOMotorClient(
                settings.mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                maxPoolSize=10,                 # Limit connections
                retryWrites=True
            )
            self.db = self.client[settings.database_name]
            self.documents_collection = self.db.documents
            self.evaluations_collection = self.db.evaluations

            # Test connection with timeout
            await self.client.admin.command("ping")
            await self._setup_indexes()
            self._mongo_connected = True
            logger.info(f"Connected to MongoDB: {settings.database_name}")
            
            # Connect to Redis
            await self.redis_manager.connect()
            logger.info("Connected to Redis")

            self._is_connected = True
            logger.info("DatabaseManager fully connected (MongoDB + Redis)")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    async def disconnect(self):
        """Disconnect from both MongoDB and Redis"""
        try:
            if self.client:
                # Properly close MongoDB connection
                self.client.close()
                # Wait for the client to close properly
                await asyncio.sleep(0.1)
                self._mongo_connected = False
                logger.info("Disconnected from MongoDB")
        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {e}")
            
        try:
            await self.redis_manager.disconnect()
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
        
        self._is_connected = False
        logger.info("DatabaseManager fully disconnected")

    async def _setup_indexes(self):
        """Setup MongoDB indexes"""
        try:
            await self.documents_collection.create_indexes([
                IndexModel([("content", TEXT)], name="content_text_index"),
                IndexModel([("metadata.category", 1)], name="category_index"),
                IndexModel([("metadata.intent", 1)], name="intent_index"),
                IndexModel([("metadata.source", 1)], name="source_index"),
                IndexModel([("document_type", 1)], name="document_type_index"),
                IndexModel([("created_at", -1)], name="created_at_index"),
            ])

            # Note: No conversation indexes needed since using Redis

            await self.evaluations_collection.create_indexes([
                IndexModel([("created_at", -1)], name="eval_created_at_index"),
                IndexModel([("overall_score", -1)], name="eval_score_index"),
            ])

            logger.info("MongoDB indexes set up successfully")

        except Exception as e:
            logger.error(f"Index setup failed: {e}")
            raise

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[Any]:
        """Store documents into the collection"""
        if not documents:
            return []

        try:
            # Clear all documents (can be made optional in production)
            await self.documents_collection.delete_many({})

            result = await self.documents_collection.insert_many(documents)
            logger.info(f"{len(result.inserted_ids)} documents stored")
            return result.inserted_ids

        except Exception as e:
            logger.error(f"Document storage error: {e}")
            raise

    async def get_documents_count(self) -> int:
        try:
            return await self.documents_collection.count_documents({})
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    async def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Cosine similarity search using stored embeddings"""
        try:
            cursor = self.documents_collection.find({"embedding": {"$exists": True}})
            documents = await cursor.to_list(length=None)

            if not documents:
                return []

            similarities = []
            for doc in documents:
                emb = doc.get("embedding")
                if emb:
                    similarity = cosine_similarity([query_embedding], [emb])[0][0]
                    doc["similarity"] = similarity
                    similarities.append(doc)

            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    async def text_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """MongoDB text search"""
        try:
            cursor = self.documents_collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

            return await cursor.to_list(length=top_k)

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search: combines text and vector-based search"""
        try:
            text_results = await self.text_search(query, top_k)
            vector_results = await self.similarity_search(query_embedding, top_k)

            seen_ids = set()
            combined_results = []

            for doc in vector_results + text_results:
                doc_id = str(doc.get("_id"))
                if doc_id not in seen_ids:
                    combined_results.append(doc)
                    seen_ids.add(doc_id)

            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def save_conversation(
        self,
        user_id: str,  # Changed from session_id to user_id
        user_query: str,
        response: str,
        retrieved_docs: List[Dict],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store a user conversation in Redis"""
        try:
            success = await self.redis_manager.save_conversation(
                user_id=user_id,
                user_query=user_query,
                response=response,
                retrieved_docs=retrieved_docs,
                metadata=metadata
            )
            
            if success:
                logger.debug(f"Saved conversation to Redis: {user_id}")
            else:
                logger.error(f"Failed to save conversation to Redis: {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise

    async def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Fetch past conversation history from Redis"""
        try:
            history = await self.redis_manager.get_conversation_history(user_id, limit)
            logger.debug(f"Retrieved {len(history)} conversations from Redis for user: {user_id}")
            return history

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def get_conversations_count(self) -> int:
        """Get total conversation count across all users in Redis"""
        try:
            user_sessions = await self.redis_manager.get_all_user_sessions()
            total_count = 0
            
            for user_id in user_sessions:
                count = await self.redis_manager.get_conversation_count(user_id)
                total_count += count
                
            return total_count
        except Exception as e:
            logger.error(f"Failed to count conversations: {e}")
            return 0

    async def save_evaluation(self, evaluation_data: Dict[str, Any]):
        """Save evaluation results"""
        try:
            evaluation_data["created_at"] = datetime.now(timezone.utc)
            await self.evaluations_collection.insert_one(evaluation_data)
            logger.info("Evaluation saved")

        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
            raise

    async def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        """Return the most recent evaluation entry"""
        try:
            cursor = self.evaluations_collection.find().sort("created_at", -1).limit(1)
            evaluations = await cursor.to_list(length=1)
            return evaluations[0] if evaluations else None

        except Exception as e:
            logger.error(f"Failed to get latest evaluation: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint for both MongoDB and Redis"""
        try:
            # Check MongoDB with timeout
            if not self.client:
                raise Exception("MongoDB client not initialized")
                
            await asyncio.wait_for(
                self.client.admin.command("ping"), 
                timeout=5.0
            )
            mongo_status = "healthy"
            
            # Check Redis
            try:
                redis_health = await self.redis_manager.health_check()
                redis_status = redis_health.get("status", "unhealthy")
            except Exception as redis_error:
                logger.error(f"Redis health check failed: {redis_error}")
                redis_status = "unhealthy"
                redis_health = {"status": "unhealthy", "error": str(redis_error)}
            
            overall_status = "healthy" if mongo_status == "healthy" and redis_status == "healthy" else "unhealthy"
            
            return {
                "status": overall_status,
                "mongodb_status": mongo_status,
                "redis_status": redis_status,
                "documents_count": await self.get_documents_count(),
                "conversations_count": await self.get_conversations_count(),
                "last_check": datetime.now(timezone.utc)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "mongodb_status": "unknown",
                "redis_status": "unknown",
                "last_check": datetime.now(timezone.utc)
            }

    def is_connected(self) -> bool:
        return self._is_connected


# Global instance
db_manager = DatabaseManager()


async def get_database():
    """FastAPI/async dependency"""
    if not db_manager.is_connected():
        await db_manager.connect()
    return db_manager


async def init_database():
    await db_manager.connect()


async def close_database():
    await db_manager.disconnect()
