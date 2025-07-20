"""
Enhanced Ingestion service for the Impacteers RAG system
"""

import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timezone
import asyncio
import uuid
import time
import os
from sentence_transformers import SentenceTransformer
from langchain_google_vertexai import ChatVertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from embedding_service import SharedEmbeddingService

# Import PDF processing libraries (prioritize PyPDF2 since it's available)
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    PyPDF2 = None

try:
    import fitz  # PyMuPDF - fallback option
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

from config import settings
from models import DocumentInput, Document
from database import DatabaseManager

logger = logging.getLogger(__name__)


class IngestionState(TypedDict):
    documents: List[DocumentInput]
    processed_chunks: List[Dict[str, Any]]
    stored_count: int
    error: str
    stage: str
    thread_id: str
    thread_ts: float


class DocumentProcessor:
    def __init__(self):
        # Configure text splitter for better semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
            length_function=len,
        )
        
        # Log available PDF processing libraries
        if HAS_PYPDF2:
            logger.info("PyPDF2 available for PDF processing")
        elif HAS_PYMUPDF:
            logger.info("PyMuPDF (fitz) available for PDF processing")
        else:
            logger.warning("No PDF processing libraries available")

    def process_documents(self, documents: List[DocumentInput]) -> List[Dict[str, Any]]:
        """Process documents based on their type"""
        processed_docs = []
        
        for doc_input in documents:
            try:
                # All document types now processed through the unified method
                processed = self._process_document(doc_input)
                processed_docs.extend(processed)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_input.content}: {e}")
                # Continue with other documents
                continue
        
        return processed_docs

    def _process_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Unified document processor that handles all document types"""
        try:
            # Check if content is a file path
            if os.path.isfile(doc_input.content):
                # Extract text from file
                text_content = self._extract_pdf_text(doc_input.content)
                filename = os.path.basename(doc_input.content)
                logger.info(f"Extracted text from file: {filename}")
            else:
                # Treat content as raw text
                text_content = doc_input.content
                filename = doc_input.metadata.get("filename", "text_content")
                logger.info(f"Processing raw text content")
            
            return [{
                "content": text_content,
                "metadata": {
                    **doc_input.metadata,
                    "category": doc_input.category,
                    "source": "document",
                    "filename": filename,
                    "document_type": str(doc_input.document_type)
                }
            }]
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return []

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using available libraries"""
        try:
            text_content = ""
            
            # Try PyPDF2 first since it's available
            if HAS_PYPDF2:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text_content += page.extract_text()
                            text_content += "\n\n"
                    logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
                except Exception as e:
                    logger.error(f"PyPDF2 extraction failed: {e}")
                    text_content = ""
            
            # Fallback to PyMuPDF if PyPDF2 fails
            if not text_content and HAS_PYMUPDF:
                try:
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text_content += page.get_text()
                        text_content += "\n\n"  # Add page breaks
                    doc.close()
                    logger.info(f"Successfully extracted text using PyMuPDF from {pdf_path}")
                except Exception as e:
                    logger.error(f"PyMuPDF extraction failed: {e}")
                    text_content = ""
            
            if not text_content:
                # If both methods fail, raise an error
                raise ValueError(f"Could not extract text from {pdf_path}. No PDF processing libraries available.")
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise

    def create_overlapping_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create overlapping chunks using RecursiveCharacterTextSplitter"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            
            # Skip if content is too short
            if len(content.strip()) < 50:
                logger.warning(f"Document content too short, skipping: {content[:50]}...")
                continue
            
            # Use the text splitter to create overlapping chunks
            text_chunks = self.text_splitter.split_text(content)
            
            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 20:
                    continue
                    
                chunk = {
                    "content": chunk_text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "chunk_length": len(chunk_text.strip()),
                        "chunking_method": "overlapping"
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} overlapping chunks")
        return chunks


class IngestionService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.llm = ChatVertexAI(
            model=settings.llm_model,
            project=settings.project_id,
            location=settings.location,
            temperature=settings.llm_temperature,
            model_kwargs={"convert_system_message_to_human": True},
        )
        self.embedding_model = SharedEmbeddingService.get_instance()
        self.processor = DocumentProcessor()
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the ingestion workflow graph"""
        workflow = StateGraph(IngestionState)
        
        # Add nodes
        workflow.add_node("process_documents", self._process_documents)
        workflow.add_node("create_embeddings", self._create_embeddings)
        workflow.add_node("store_documents", self._store_documents)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("process_documents")
        
        # Add edges
        workflow.add_conditional_edges(
            "process_documents",
            self._should_handle_error,
            {"error": "handle_error", "continue": "create_embeddings"}
        )
        workflow.add_conditional_edges(
            "create_embeddings",
            self._should_handle_error,
            {"error": "handle_error", "continue": "store_documents"}
        )
        workflow.add_conditional_edges(
            "store_documents",
            self._should_handle_error,
            {"error": "handle_error", "continue": END}
        )
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.memory)

    def _should_handle_error(self, state: IngestionState) -> str:
        """Check if we should handle an error"""
        return "error" if state.get("error") else "continue"

    async def _process_documents(self, state: IngestionState) -> IngestionState:
        """Process documents and create chunks"""
        try:
            state["stage"] = "processing"
            processed_docs = self.processor.process_documents(state["documents"])
            
            # Use semantic chunking for better results
            state["processed_chunks"] = self.processor.create_overlapping_chunks(processed_docs)
            
            # Remove original documents to save memory
            state.pop("documents", None)
            
            logger.info(f"Processed {len(processed_docs)} documents into {len(state['processed_chunks'])} chunks")
            return state
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            state["error"] = f"Document processing failed: {e}"
            return state

    async def _create_embeddings(self, state: IngestionState) -> IngestionState:
        """Create embeddings for chunks"""
        try:
            state["stage"] = "embedding"
            chunks = state["processed_chunks"]
            
            if not chunks:
                logger.warning("No chunks to embed")
                return state
            
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [chunk["content"] for chunk in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(
                    texts, 
                    show_progress_bar=True,
                    batch_size=min(batch_size, len(texts))
                )
                
                # Add embeddings and IDs to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    chunk["doc_id"] = f"doc_{uuid.uuid4()}"
                    chunk["created_at"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Created embeddings for {len(chunks)} chunks")
            return state
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            state["error"] = f"Embedding creation failed: {e}"
            return state

    async def _store_documents(self, state: IngestionState) -> IngestionState:
        """Store processed chunks in database"""
        try:
            state["stage"] = "storing"
            chunks = state["processed_chunks"]
            
            if not chunks:
                logger.warning("No chunks to store")
                state["stored_count"] = 0
                return state
            
            # Store in database
            inserted_ids = await self.db_manager.store_documents(chunks)
            state["stored_count"] = len(inserted_ids)
            
            # Clean up chunks for checkpointing
            for chunk in chunks:
                chunk.pop("_id", None)
                if isinstance(chunk.get("created_at"), datetime):
                    chunk["created_at"] = chunk["created_at"].isoformat()
            
            logger.info(f"Stored {state['stored_count']} chunks in database")
            return state
            
        except Exception as e:
            logger.error(f"Document storage failed: {e}")
            state["error"] = f"Document storage failed: {e}"
            return state

    async def _handle_error(self, state: IngestionState) -> IngestionState:
        """Handle errors in the pipeline"""
        logger.error(f"Pipeline error in stage {state.get('stage', 'unknown')}: {state.get('error', 'unknown')}")
        return state

    async def ingest_documents(self, documents: List[DocumentInput]) -> Dict[str, Any]:
        """Main ingestion method with consistent thread management"""
        start_time = time.time()
        
        # Use content-based thread ID for ingestion consistency
        content_hash = hash(str([doc.content for doc in documents]))
        thread_id = f"ingestion_{abs(content_hash)}"  # Consistent for same content
        
        initial_state = IngestionState(
            documents=documents,
            processed_chunks=[],
            stored_count=0,
            error="",
            stage="initialized",
            thread_id=thread_id,
            thread_ts=time.time()
        )
        
        # Use consistent config for memory persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            processing_time = time.time() - start_time
            
            return {
                "success": not final_state.get("error"),
                "error": final_state.get("error", ""),
                "documents_processed": len(documents),
                "chunks_created": len(final_state.get("processed_chunks", [])),
                "stored_count": final_state.get("stored_count", 0),
                "processing_time": processing_time,
                "stage": final_state.get("stage", "unknown"),
                "thread_id": thread_id
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ingestion failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "documents_processed": len(documents),
                "chunks_created": 0,
                "stored_count": 0,
                "processing_time": processing_time,
                "stage": "failed",
                "thread_id": thread_id
            }

    async def get_sample_documents(self) -> List[DocumentInput]:
        """Get sample documents for testing - now uses actual PDF files"""
        return [
            DocumentInput(
                content="feature.pdf", 
                document_type="manual", 
                category="platform_features", 
                metadata={"source": "knowledge_base", "filename": "feature.pdf"}
            ),
            DocumentInput(
                content="updated_rag_knowledge.pdf", 
                document_type="manual", 
                category="job_search_opportunities", 
                metadata={"source": "knowledge_base", "filename": "updated_rag_knowledge.pdf"}
            ),
            DocumentInput(
                content="update_knowledge.pdf", 
                document_type="manual", 
                category="platform_features", 
                metadata={"source": "knowledge_base", "filename": "update_knowledge.pdf"}
            )
        ]