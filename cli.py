#!/usr/bin/env python3
"""
Command Line Interface for the Impacteers RAG system
"""

import asyncio
import uuid
import logging
from typing import Dict, Any
import click

from config import settings
from database import init_database, close_database, get_database
from ingestion_service import IngestionService
from inference_service import InferenceService
from evaluation_service import EvaluationService
from models import ChatRequest, DocumentInput

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystemCLI:
    """Command-line interface for the RAG system"""
    
    def __init__(self):
        self.db_manager = None
        self.ingestion_service = None
        self.inference_service = None
        self.evaluation_service = None
        self.session_id = f"cli_session_{uuid.uuid4()}"
    
    async def initialize(self):
        """Initialize all services"""
        try:
            # Initialize database
            await init_database()
            self.db_manager = await get_database()
            
            # Initialize services
            self.ingestion_service = IngestionService(self.db_manager)
            self.inference_service = InferenceService(self.db_manager)
            self.evaluation_service = EvaluationService(self.db_manager, self.inference_service)
            
            logger.info("CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"CLI initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        await close_database()
        logger.info("CLI cleanup completed")
    
    async def setup_system(self) -> Dict[str, Any]:
        """Setup the system with sample data"""
        click.echo("🔄 Setting up RAG system...")
        
        # Get sample documents
        sample_docs = await self.ingestion_service.get_sample_documents()
        
        # Ingest documents
        click.echo("📚 Ingesting sample documents...")
        ingestion_result = await self.ingestion_service.ingest_documents(sample_docs)
        
        if not ingestion_result["success"]:
            click.echo(f"❌ Ingestion failed: {ingestion_result['error']}")
            return {"success": False, "error": ingestion_result["error"]}
        
        # Test inference
        click.echo("🧪 Testing inference...")
        test_request = ChatRequest(query="I'm looking for a job")
        inference_result = await self.inference_service.chat(test_request)
        
        # Run evaluation
        click.echo("📊 Running evaluation...")
        evaluation_result = await self.evaluation_service.run_evaluation()
        
        if not evaluation_result.success:
            click.echo(f"❌ Evaluation failed: {evaluation_result.error}")
            return {"success": False, "error": evaluation_result.error}
        
        click.echo("✅ System setup completed!")
        click.echo(f"📄 Documents processed: {ingestion_result['documents_processed']}")
        click.echo(f"🧩 Chunks created: {ingestion_result['chunks_created']}")
        click.echo(f"📈 System score: {evaluation_result.overall_score:.3f}")
        
        return {
            "success": True,
            "ingestion": ingestion_result,
            "inference": inference_result,
            "evaluation": evaluation_result
        }
    
    async def test_memory_functionality(self):
        """Test memory functionality"""
        click.echo("🧪 Testing memory functionality...")
        
        # Test with same session ID
        session_id = "test_session_123"
        
        click.echo(f"Testing with session: {session_id}")
        
        # First message
        click.echo("\n--- First Message ---")
        request1 = ChatRequest(query="Hello, my name is John", session_id=session_id)
        response1 = await self.inference_service.chat(request1)
        click.echo(f"Response 1: {response1.response[:100]}...")
        
        # Check database immediately
        click.echo("\n--- Checking Database ---")
        await self.debug_database_conversations(session_id)
        
        # Wait a moment for database write
        await asyncio.sleep(1)
        
        # Second message - should remember first
        click.echo("\n--- Second Message ---")
        request2 = ChatRequest(query="What's my name?", session_id=session_id)
        response2 = await self.inference_service.chat(request2)
        click.echo(f"Response 2: {response2.response[:100]}...")
        
        # Third message - test conversation history
        click.echo("\n--- Third Message ---")
        request3 = ChatRequest(query="What have we discussed so far?", session_id=session_id)
        response3 = await self.inference_service.chat(request3)
        click.echo(f"Response 3: {response3.response[:200]}...")
        
        click.echo("✅ Memory test completed")
    
    async def debug_database_conversations(self, session_id: str):
        """Debug method to check what's actually in the database"""
        try:
            # Check total conversations in database
            total_convs = await self.db_manager.conversations_collection.count_documents({})
            click.echo(f"Total conversations in database: {total_convs}")
            
            # Check conversations for this session
            session_convs = await self.db_manager.conversations_collection.count_documents({
                "session_id": session_id
            })
            click.echo(f"Conversations for session {session_id}: {session_convs}")
            
            # Get all conversations (limit 5 for debugging)
            all_convs = await self.db_manager.conversations_collection.find({}).limit(5).to_list(5)
            click.echo("Sample conversations in database:")
            for conv in all_convs:
                session = conv.get('session_id', 'N/A')
                query = conv.get('user_query', 'N/A')
                click.echo(f"  Session: {session}, Query: {query}")
                
            # Check for exact session match
            exact_match = await self.db_manager.conversations_collection.find({
                "session_id": session_id
            }).to_list(10)
            click.echo(f"Exact matches for {session_id}: {len(exact_match)}")
            
        except Exception as e:
            click.echo(f"❌ Database debug failed: {e}")
    
    async def interactive_chat(self):
        """Interactive chat mode"""
        click.echo("💬 Interactive Chat Mode")
        click.echo("Type 'quit' to exit, 'help' for commands")
        click.echo("-" * 50)
        
        while True:
            try:
                user_input = click.prompt("You", type=str).strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'status':
                    await self._show_status()
                    continue
                
                if user_input.lower() == 'history':
                    await self._show_history()
                    continue
                
                if user_input.lower() == 'debug':
                    await self.debug_database_conversations(self.session_id)
                    continue
                
                if not user_input:
                    continue
                
                # Process chat
                click.echo("🤖 Assistant: ", nl=False)
                
                request = ChatRequest(query=user_input, session_id=self.session_id)
                response = await self.inference_service.chat(request)
                
                click.echo(response.response)
                
                # Show debug info
                if response.retrieved_docs > 0:
                    click.echo(f"   📚 Retrieved {response.retrieved_docs} documents")
                
                if response.error:
                    click.echo(f"   ⚠️  Error: {response.error}")
                
            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"❌ Error: {e}")
    
    async def batch_test(self):
        """Run batch test with predefined queries"""
        click.echo("🧪 Running batch test...")
        
        test_queries = [
            "I'm looking for a job",
            "What courses do you offer?",
            "How can I assess my skills?",
            "Tell me about mentorship",
            "What's IIPL?",
            "How do I improve my resume?"
        ]
        
        session_id = f"batch_test_{uuid.uuid4()}"
        
        for i, query in enumerate(test_queries, 1):
            click.echo(f"\n[{i}/{len(test_queries)}] Query: {query}")
            
            request = ChatRequest(query=query, session_id=session_id)
            response = await self.inference_service.chat(request)
            
            click.echo(f"Response: {response.response}")
            click.echo(f"Retrieved docs: {response.retrieved_docs}")
            click.echo(f"Processing time: {response.processing_time:.2f}s")
        
        click.echo("\n✅ Batch test completed!")
    
    async def run_evaluation(self):
        """Run system evaluation"""
        click.echo("📊 Running system evaluation...")
        
        result = await self.evaluation_service.run_evaluation()
        
        if result.success:
            click.echo("✅ Evaluation completed!")
            click.echo(f"📈 Overall Score: {result.overall_score:.3f}")
            click.echo(f"🎯 Test Cases: {result.test_cases_count}")
            click.echo(f"⏱️  Processing Time: {result.processing_time:.2f}s")
            
            # Show detailed metrics
            click.echo("\n📊 Detailed Metrics:")
            for metric, value in result.retrieval_metrics.items():
                click.echo(f"  {metric}: {value:.3f}")
            
            for metric, value in result.generation_metrics.items():
                click.echo(f"  {metric}: {value:.3f}")
            
            # Show report
            if result.evaluation_report:
                click.echo("\n📄 Evaluation Report:")
                click.echo("-" * 50)
                click.echo(result.evaluation_report)
        else:
            click.echo(f"❌ Evaluation failed: {result.error}")
    
    def _show_help(self):
        """Show help commands"""
        help_text = """
Available commands:
  help     - Show this help message
  status   - Show system status
  history  - Show conversation history
  debug    - Debug database conversations
  quit     - Exit the chat

Just type your question to chat with the assistant!
        """
        click.echo(help_text)
    
    async def _show_status(self):
        """Show system status"""
        try:
            health_status = await self.db_manager.health_check()
            
            click.echo("📊 System Status:")
            click.echo(f"  Health: {health_status['status']}")
            click.echo(f"  Documents: {health_status['documents_count']}")
            click.echo(f"  Conversations: {health_status['conversations_count']}")
            
        except Exception as e:
            click.echo(f"❌ Status check failed: {e}")
    
    async def _show_history(self):
        """Show conversation history"""
        try:
            history = await self.inference_service.get_conversation_history(self.session_id)
            
            if not history:
                click.echo("No conversation history found.")
                return
            
            click.echo("💬 Conversation History:")
            for i, conv in enumerate(history, 1):
                click.echo(f"  [{i}] You: {conv['user_query']}")
                click.echo(f"      Bot: {conv['response'][:100]}...")
                click.echo()
        
        except Exception as e:
            click.echo(f"❌ History fetch failed: {e}")


# CLI Commands
@click.group()
def cli():
    """Impacteers RAG System CLI"""
    pass


@cli.command()
def setup():
    """Setup the RAG system with sample data"""
    async def _setup():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            await rag_cli.setup_system()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_setup())


@cli.command()
def test_memory():
    """Test memory functionality"""
    async def _test_memory():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            await rag_cli.test_memory_functionality()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_test_memory())


@cli.command()
def debug_db():
    """Debug database conversations"""
    async def _debug_db():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            session_id = click.prompt("Enter session ID to debug", default="test_session_123")
            await rag_cli.debug_database_conversations(session_id)
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_debug_db())


@cli.command()
def chat():
    """Start interactive chat mode"""
    async def _chat():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            
            # Check if system is set up
            docs_count = await rag_cli.db_manager.get_documents_count()
            if docs_count == 0:
                click.echo("⚠️  No documents found. Running setup first...")
                await rag_cli.setup_system()
            
            await rag_cli.interactive_chat()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_chat())


@cli.command()
def test():
    """Run batch test with predefined queries"""
    async def _test():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            await rag_cli.batch_test()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_test())


@cli.command()
def evaluate():
    """Run system evaluation"""
    async def _evaluate():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            await rag_cli.run_evaluation()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_evaluate())


@cli.command()
def status():
    """Show system status"""
    async def _status():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            await rag_cli._show_status()
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_status())


@cli.command()
def full_demo():
    """Run complete demo (setup + chat + evaluation)"""
    async def _full_demo():
        rag_cli = RAGSystemCLI()
        try:
            await rag_cli.initialize()
            
            # Setup
            click.echo("🚀 Starting full demo...")
            setup_result = await rag_cli.setup_system()
            
            if not setup_result["success"]:
                click.echo(f"❌ Setup failed: {setup_result['error']}")
                return
            
            # Batch test
            click.echo("\n🧪 Running batch test...")
            await rag_cli.batch_test()
            
            # Evaluation
            click.echo("\n📊 Running evaluation...")
            await rag_cli.run_evaluation()
            
            # Offer interactive chat
            if click.confirm("\nWould you like to start interactive chat?"):
                await rag_cli.interactive_chat()
            
        finally:
            await rag_cli.cleanup()
    
    asyncio.run(_full_demo())


if __name__ == "__main__":
    cli()