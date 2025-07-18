"""
Database operations for the Impacteers RAG system
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, TEXT
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import settings
from models import Document

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async MongoDB database manager"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.documents_collection = None
        self.conversations_collection = None
        self.evaluations_collection = None
        self._is_connected = False

    async def connect(self):
        """Connect to MongoDB"""
        if self._is_connected:
            return  # Already connected

        try:
            self.client = AsyncIOMotorClient(settings.mongo_uri)
            self.db = self.client[settings.database_name]
            self.documents_collection = self.db.documents
            self.conversations_collection = self.db.conversations
            self.evaluations_collection = self.db.evaluations

            await self.client.admin.command("ping")  # Check connection
            await self._setup_indexes()

            self._is_connected = True
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self._is_connected = False
            logger.info("Disconnected from MongoDB")

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

            await self.conversations_collection.create_indexes([
                IndexModel([("session_id", 1)], name="session_id_index"),
                IndexModel([("timestamp", -1)], name="timestamp_index"),
                IndexModel([("session_id", 1), ("timestamp", -1)], name="session_timestamp_index"),
            ])

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
        session_id: str,
        user_query: str,
        response: str,
        retrieved_docs: List[Dict],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store a user conversation"""
        try:
            conversation = {
                "session_id": session_id,
                "user_query": user_query,
                "response": response,
                "retrieved_docs_count": len(retrieved_docs),
                "retrieved_docs": retrieved_docs,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow()
            }

            await self.conversations_collection.insert_one(conversation)
            logger.debug(f"Saved conversation: {session_id}")

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise

    async def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Fetch past conversation history"""
        try:
            cursor = self.conversations_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)

            return list(reversed(await cursor.to_list(length=limit)))

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def get_conversations_count(self) -> int:
        try:
            return await self.conversations_collection.count_documents({})
        except Exception as e:
            logger.error(f"Failed to count conversations: {e}")
            return 0

    async def save_evaluation(self, evaluation_data: Dict[str, Any]):
        """Save evaluation results"""
        try:
            evaluation_data["created_at"] = datetime.utcnow()
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
        """Health check endpoint for MongoDB"""
        try:
            await self.client.admin.command("ping")
            return {
                "status": "healthy",
                "documents_count": await self.get_documents_count(),
                "conversations_count": await self.get_conversations_count(),
                "last_check": datetime.utcnow()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow()
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
