"""
Redis manager for conversation storage in the Impacteers RAG system
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Async Redis manager for conversation storage"""

    def __init__(self):
        self.client: Optional[aioredis.Redis] = None
        self._is_connected = False
        self.conversation_ttl = 86400  # 1 day in seconds
        self._connection_lock = None

    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):  # General objects
            return str(obj)
        else:
            return str(obj)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        return False  # Don't suppress exceptions

    async def connect(self):
        """Connect to Redis with thread safety"""
        if self._is_connected:
            return  # Already connected

        # Initialize lock if not exists
        if self._connection_lock is None:
            self._connection_lock = asyncio.Lock()

        async with self._connection_lock:
            # Double-check after acquiring lock
            if self._is_connected:
                return

            try:
                # Create Redis connection - prefer URL if available
                if settings.redis_url:
                    # Use URL connection string
                    self.client = aioredis.from_url(
                        settings.redis_url,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30,
                        max_connections=20,
                        retry_on_error=[ConnectionError, TimeoutError]
                    )
                else:
                    # Use individual parameters
                    self.client = aioredis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        db=settings.redis_db,
                        username=settings.redis_username,
                        password=settings.redis_password,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30,
                        max_connections=20,
                        retry_on_error=[ConnectionError, TimeoutError]
                    )

                # Test connection
                await self.client.ping()
                self._is_connected = True
                logger.info("Connected to Redis successfully")
            except Exception as e:
                logger.error(f"Redis connection error: {e}")
                raise

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client and self._is_connected:
            await self.client.aclose()
            self._is_connected = False
            logger.info("Disconnected from Redis")

    async def save_conversation(
        self, 
        user_id: str, 
        user_query: str, 
        response: str, 
        retrieved_docs: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Save conversation to Redis with TTL"""
        try:
            if not self._is_connected:
                await self.connect()

            conversation = {
                "user_query": user_query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrieved_docs": retrieved_docs or [],
                "metadata": metadata or {}
            }

            # Use user_id as the key prefix for conversations
            conversation_key = f"conversations:{user_id}"
            
            # Get existing conversations
            existing_conversations = await self.get_conversation_history(user_id)
            
            # Add new conversation
            existing_conversations.append(conversation)
            
            # Keep only last 50 conversations to prevent memory issues
            if len(existing_conversations) > 50:
                existing_conversations = existing_conversations[-50:]
            
            # Save back to Redis with TTL
            try:
                serialized_data = json.dumps(existing_conversations, default=self._json_serializer)
            except (TypeError, ValueError) as e:
                logger.error(f"JSON serialization failed: {e}")
                return False
                
            await self.client.setex(
                conversation_key,
                self.conversation_ttl,
                serialized_data
            )
            
            logger.info(f"Saved conversation for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation to Redis: {e}")
            return False

    async def get_conversation_history(
        self, 
        user_id: str, 
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history from Redis"""
        try:
            if not self._is_connected:
                await self.connect()

            conversation_key = f"conversations:{user_id}"
            conversations_json = await self.client.get(conversation_key)
            
            if not conversations_json:
                return []
            
            try:
                conversations = json.loads(conversations_json)
                if not isinstance(conversations, list):
                    logger.error(f"Conversations data is not a list: {type(conversations)}")
                    return []
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse conversations JSON: {e}")
                return []
            
            # Apply limit if specified
            if limit:
                conversations = conversations[-limit:]
            
            logger.info(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get conversation history from Redis: {e}")
            return []

    async def delete_conversation_history(self, user_id: str) -> bool:
        """Delete conversation history for a user"""
        try:
            if not self._is_connected:
                await self.connect()

            conversation_key = f"conversations:{user_id}"
            result = await self.client.delete(conversation_key)
            
            logger.info(f"Deleted conversation history for user {user_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete conversation history from Redis: {e}")
            return False

    async def get_conversation_count(self, user_id: str) -> int:
        """Get count of conversations for a user"""
        try:
            conversations = await self.get_conversation_history(user_id)
            return len(conversations)
        except Exception as e:
            logger.error(f"Failed to get conversation count from Redis: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Health check for Redis connection"""
        try:
            if not self._is_connected:
                await self.connect()

            # Test basic operations
            test_key = "health_check"
            await self.client.setex(test_key, 60, "test")
            test_value = await self.client.get(test_key)
            await self.client.delete(test_key)
            
            return {
                "status": "healthy" if test_value == "test" else "unhealthy",
                "connected": self._is_connected,
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }

    async def get_all_user_sessions(self) -> List[str]:
        """Get all user sessions (for admin purposes)"""
        try:
            if not self._is_connected:
                await self.connect()

            # Get all conversation keys
            keys = await self.client.keys("conversations:*")
            user_ids = [key.split(":", 1)[1] for key in keys]
            
            return user_ids
            
        except Exception as e:
            logger.error(f"Failed to get user sessions from Redis: {e}")
            return []

    async def extend_conversation_ttl(self, user_id: str) -> bool:
        """Extend TTL for user conversations (useful for active users)"""
        try:
            if not self._is_connected:
                await self.connect()

            conversation_key = f"conversations:{user_id}"
            result = await self.client.expire(conversation_key, self.conversation_ttl)
            
            if result:
                logger.info(f"Extended TTL for user {user_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extend TTL for user {user_id}: {e}")
            return False