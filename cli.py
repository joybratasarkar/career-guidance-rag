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