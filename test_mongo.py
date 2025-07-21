#!/usr/bin/env python3
"""
Test MongoDB connection for debugging deployment issues
"""
import asyncio
import os
import sys
from config import settings

async def test_mongodb():
    """Test MongoDB connection"""
    print(f"Testing MongoDB connection...")
    print(f"MongoDB URI: {settings.mongo_uri[:50]}..." if len(settings.mongo_uri) > 50 else f"MongoDB URI: {settings.mongo_uri}")
    print(f"Database: {settings.database_name}")
    
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        
        client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=10000,  # 10 second timeout
            connectTimeoutMS=15000,          # 15 second connection timeout
        )
        
        # Test connection
        await client.admin.command("ping")
        print("✅ MongoDB connection successful!")
        
        # Test database access
        db = client[settings.database_name]
        collections = await db.list_collection_names()
        print(f"✅ Database accessible. Collections: {collections}")
        
        client.close()
        print("✅ MongoDB connection closed cleanly")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mongodb())
    sys.exit(0 if success else 1)