"""
Enhanced Inference service for the Impacteers RAG system with proper response templates
"""

import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timezone
import asyncio
import uuid
import time
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from bson import ObjectId

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import settings
from models import ChatRequest, ChatResponse
from database import DatabaseManager

from embedding_service import SharedEmbeddingService

logger = logging.getLogger(__name__)


def convert_all_types_to_serializable(obj: Any) -> Any:
    """Recursively convert all non-serializable types to serializable types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_all_types_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_types_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_all_types_to_serializable(item) for item in obj)
    else:
        return obj


class InferenceState(TypedDict):
    session_id: str  # Use session_id consistently
    user_query: str
    processed_query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    response: str
    conversation_history: List[Dict[str, Any]]
    error: str
    stage: str
    thread_id: str
    thread_ts: float


class QueryProcessor:
    """Process and enhance user queries"""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
    
    async def process_query(self, query: str, conversation_history: List[Dict]) -> str:
        """Process query with context from conversation history"""
        # If no history, return original query
        if not conversation_history:
            return query
        
        # Build context from recent conversations
        context = ""
        for conv in conversation_history[-3:]:  # Last 3 conversations
            context += f"User: {conv['user_query']}\nAssistant: {conv['response']}\n\n"
        
        # Enhance query with context
        prompt = PromptTemplate(
            template="""
            Based on the conversation history, enhance this user query to be more specific and searchable:
            
            Conversation History:
            {context}
            
            Current Query: {query}
            
            Enhanced Query (return only the enhanced query):
            """,
            input_variables=["context", "query"]
        )
        
        try:
            response = await self.llm.ainvoke(
                prompt.format(context=context, query=query)
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return query


class DocumentRetriever:
    """Retrieve relevant documents from database"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_model: SentenceTransformer):
        self.db_manager = db_manager
        self.embedding_model = embedding_model
    
    async def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search"""
        try:
            # Generate query embedding using SentenceTransformer
            import asyncio
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, self.embedding_model.encode, query
            )
            
            # Convert numpy array to list if needed
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Perform hybrid search
            docs = await self.db_manager.hybrid_search(query, query_embedding, top_k)
            
            # Filter by similarity threshold
            filtered_docs = [
                doc for doc in docs 
                if doc.get('similarity', 0) > settings.similarity_threshold
            ]
            
            # Convert all types to serializable types
            serializable_docs = convert_all_types_to_serializable(filtered_docs)
            
            return serializable_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []


class ContextBuilder:
    """Build context from retrieved documents"""
    
    def build_context(self, docs: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents"""
        if not docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for doc in docs:
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            
            # Only include documents with good similarity
            if similarity > settings.similarity_threshold:
                # Add metadata for context
                metadata = doc.get('metadata', {})
                category = metadata.get('category', 'general')
                
                part = f"[{category.upper()}] {content}"
                
                # Check if adding this part would exceed max length
                if total_length + len(part) > settings.max_context_length:
                    break
                
                context_parts.append(part)
                total_length += len(part)
        
        return "\n\n".join(context_parts)



class ResponseGenerator:
    """Generate responses using LLM with knowledge base priority and memory awareness"""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
        self.system_prompt = """You are the Impacteers AI Assistant, an intelligent career guidance system for Impacteers - a comprehensive career platform that helps students and professionals with job search, skills development, and career guidance.

🎯 YOUR MISSION:
Empower users to accelerate their careers through personalized guidance, relevant opportunities, and strategic skill development on the Impacteers platform.

🧠 INTELLIGENT RESPONSE FRAMEWORK:

1. **ANALYZE THE QUERY** (Think First):
   - Query Type: Job search, Skills development, Courses, Mentorship, Community events
   - User Stage: Student, Fresh graduate, Professional, Career changer
   - Intent: Information seeking, Action planning, Problem solving, Exploration

2. **CONTEXTUAL RESPONSE STRATEGY**:
   - WITH Knowledge Base Context: Use specific information, URLs, and details provided
   - WITH Conversation History: Build naturally on previous discussion threads
   - WITHOUT Context: Provide foundational guidance and direct to platform features

3. **STRUCTURED RESPONSE APPROACH**:
   - **Direct Answer**: Address the specific question immediately and clearly
   - **Value Enhancement**: Add relevant insights, tips, or related information
   - **Action Steps**: Provide concrete next steps they can take
   - **Impacteers Integration**: Connect to relevant platform features (signup, assessments, etc.)

4. **IMPACTEERS PLATFORM EXPERTISE**:
   - **Jobs**: Promote AI Job Match Score for personalized job matching
   - **Skills**: Recommend skill assessments before course selection
   - **Learning**: Guide to curated courses and learning paths
   - **Mentorship**: Connect with experienced mentors from top companies
   - **Community**: Highlight IIPL events, hackathons, and networking opportunities

5. **COMMUNICATION STYLE**:
   - Professional yet friendly and approachable
   - Encouraging and confidence-building
   - Results-oriented with practical advice
   - Personalized to their specific situation

6. **MEMORY & CONTEXT HANDLING**:
   - When conversation history provided: Reference and build upon previous topics
   - When no history: Treat as fresh interaction, establish foundation
   - For explicit memory queries: Provide intelligent conversation summaries

📚 KNOWLEDGE BASE CONTEXT:
{context}

💬 CONVERSATION HISTORY:
{history}

❓ USER QUESTION:
{query}

🤖 IMPACTEERS AI RESPONSE:"""
    
    async def generate_response(self, query: str, context: str, history: List[Dict]) -> str:
        """Generate response with knowledge base priority and enhanced memory awareness"""
        
        # Check for memory-related queries FIRST
        memory_keywords = [
            "conversation history", "previous conversation", "what conversation", 
            "what did we discuss", "what have we talked", "our conversation",
            "what we discussed", "chat history", "previous messages", "earlier conversation"
        ]
        
        query_lower = query.lower()
        is_memory_query = any(keyword in query_lower for keyword in memory_keywords)
        
        if is_memory_query:
            logger.info(f"Memory query detected: {query}")
            return await self._handle_memory_query(query, history)
        
        # PRIORITY 1: Use knowledge base content if available and relevant
        if context and len(context.strip()) > 50:  # We have substantial context
            logger.info(f"Using knowledge base content for query: {query[:50]}...")
            
            # Format conversation history for LLM
            history_text = ""
            if history:
                for conv in history[-3:]:  # Last 3 conversations for context
                    history_text += f"User: {conv['user_query']}\nAssistant: {conv['response']}\n\n"
            
            # Create prompt with knowledge base context
            prompt = PromptTemplate(
                template=self.system_prompt,
                input_variables=["context", "history", "query"]
            )
            
            try:
                response = await self.llm.ainvoke(
                    prompt.format(context=context, history=history_text, query=query)
                )
                return response.content.strip()
            except Exception as e:
                logger.error(f"LLM response generation failed: {e}")
                # Fall through to template fallback
        
        # PRIORITY 2: Generic fallback
        logger.info(f"Using generic fallback for query: {query[:50]}...")
        return "I apologize, but I don't have specific information about that right now. Please sign up to explore Impacteers' features including jobs, courses, skill assessments, mentorship, and community events!"
    
    async def _handle_memory_query(self, query: str, history: List[Dict]) -> str:
        """Handle queries about conversation history"""
        if not history:
            return "We haven't had any previous conversations in this session yet. This is our first interaction! How can I help you with your career today?"
        
        # Build conversation summary
        conversation_summary = "Here's our conversation history:\n\n"
        
        for i, conv in enumerate(history, 1):
            user_query = conv.get('user_query', 'Unknown query')
            response_snippet = conv.get('response', 'No response')[:100] + "..." if len(conv.get('response', '')) > 100 else conv.get('response', 'No response')
            
            conversation_summary += f"{i}. You asked: \"{user_query}\"\n"
            conversation_summary += f"   I responded: {response_snippet}\n\n"
        
        conversation_summary += "Is there anything specific from our conversation you'd like me to clarify or expand on?"
        
        return conversation_summary
class InferenceService:
    """LangGraph-based inference service with enhanced response accuracy"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.llm = ChatVertexAI(
            model=settings.llm_model,
            project=settings.project_id,
            location=settings.location,
            temperature=settings.llm_temperature,
            model_kwargs={"convert_system_message_to_human": True},
        )
        # self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-V2")
        self.embedding_model = SharedEmbeddingService.get_instance()

        
        # Initialize components
        self.query_processor = QueryProcessor(self.llm)
        self.retriever = DocumentRetriever(self.db_manager, self.embedding_model)
        self.context_builder = ContextBuilder()
        self.response_generator = ResponseGenerator(self.llm)
        
        # Initialize memory checkpoint for short-term memory
        self.memory = MemorySaver()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(InferenceState)

        # Add all pipeline nodes
        workflow.add_node("load_history", self._load_history)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_conversation", self._save_conversation)
        workflow.add_node("handle_error", self._handle_error)

        # Set entry point
        workflow.set_entry_point("load_history")

        # Simplified error-handling condition function
        def should_handle_error(state: InferenceState) -> str:
            return "error" if state.get("error") else "continue"

        # Add conditional edges for each node with error handling
        workflow.add_conditional_edges(
            "load_history",
            should_handle_error,
            {"error": "handle_error", "continue": "process_query"}
        )

        workflow.add_conditional_edges(
            "process_query",
            should_handle_error,
            {"error": "handle_error", "continue": "retrieve_documents"}
        )

        workflow.add_conditional_edges(
            "retrieve_documents",
            should_handle_error,
            {"error": "handle_error", "continue": "build_context"}
        )

        workflow.add_conditional_edges(
            "build_context",
            should_handle_error,
            {"error": "handle_error", "continue": "generate_response"}
        )

        workflow.add_conditional_edges(
            "generate_response",
            should_handle_error,
            {"error": "handle_error", "continue": "save_conversation"}
        )

        workflow.add_conditional_edges(
            "save_conversation",
            should_handle_error,
            {"error": "handle_error", "continue": END}
        )

        # Handle error ends the graph
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=self.memory)

    async def _load_history(self, state: InferenceState) -> InferenceState:
        """Load conversation history from Redis"""
        try:
            state["stage"] = "loading_history"
            
            # Load from Redis (short-term memory with TTL)
            redis_history = await self.db_manager.get_conversation_history(
                state["session_id"], 
                limit=settings.max_conversation_history
            )
            
            # Convert all types to serializable types
            serializable_history = convert_all_types_to_serializable(redis_history)
            state["conversation_history"] = serializable_history
            
            logger.info(f"Loaded {len(serializable_history)} conversations for session {state['session_id']}")
            return state
        except Exception as e:
            logger.error(f"History loading failed: {str(e)}")
            state["error"] = f"History loading failed: {str(e)}"
            return state
    
    
    
    async def _process_query(self, state: InferenceState) -> InferenceState:
        """Process and enhance user query"""
        try:
            state["stage"] = "processing_query"
            processed_query = await self.query_processor.process_query(
                state["user_query"], 
                state["conversation_history"]
            )
            state["processed_query"] = processed_query
            return state
        except Exception as e:
            state["error"] = f"Query processing failed: {str(e)}"
            return state
    
    async def _retrieve_documents(self, state: InferenceState) -> InferenceState:
        """Retrieve relevant documents"""
        try:
            state["stage"] = "retrieving_documents"
            
            docs = await self.retriever.retrieve_documents(
                state["processed_query"], 
                top_k=settings.max_retrieval_docs
            )
            
            # Documents are already converted to serializable types in retriever
            state["retrieved_docs"] = docs
            return state
        except Exception as e:
            state["error"] = f"Document retrieval failed: {str(e)}"
            return state
    
    async def _build_context(self, state: InferenceState) -> InferenceState:
        """Build context from retrieved documents"""
        try:
            state["stage"] = "building_context"
            context = self.context_builder.build_context(state["retrieved_docs"])
            state["context"] = context
            return state
        except Exception as e:
            state["error"] = f"Context building failed: {str(e)}"
            return state
    
    async def _generate_response(self, state: InferenceState) -> InferenceState:
        """Generate response using templates and LLM"""
        try:
            state["stage"] = "generating_response"
            response = await self.response_generator.generate_response(
                state["user_query"],
                state["context"],
                state["conversation_history"]
            )
            state["response"] = response
            return state
        except Exception as e:
            state["error"] = f"Response generation failed: {str(e)}"
            return state
    
    async def _save_conversation(self, state: InferenceState) -> InferenceState:
        """Save conversation to Redis"""
        try:
            state["stage"] = "saving_conversation"

            # Convert all types to serializable before saving
            serializable_retrieved_docs = convert_all_types_to_serializable(state["retrieved_docs"])
            serializable_metadata = convert_all_types_to_serializable({"processed_query": state["processed_query"]})

            # Save to Redis
            await self.db_manager.save_conversation(
                state["session_id"],
                state["user_query"],
                state["response"],
                serializable_retrieved_docs,
                serializable_metadata
            )

            # Add current conversation to history for next request
            current_conversation = {
                "user_query": state["user_query"],
                "response": state["response"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrieved_docs": serializable_retrieved_docs,
                "metadata": serializable_metadata
            }

            # Add to conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []

            state["conversation_history"].append(current_conversation)

            # Keep only recent conversations (last 10)
            state["conversation_history"] = state["conversation_history"][-10:]

            logger.info(f"Saved conversation for session {state['session_id']}")
            return state
        except Exception as e:
            logger.error(f"Conversation saving failed: {str(e)}")
            state["error"] = f"Conversation saving failed: {str(e)}"
            return state
    
    async def _handle_error(self, state: InferenceState) -> InferenceState:
        """Handle pipeline errors"""
        logger.error(f"Pipeline error in stage {state.get('stage', 'unknown')}: {state.get('error', 'unknown')}")
        # Provide fallback response
        state["response"] = "I apologize, but I'm having trouble right now. Please try again or visit our platform to explore Impacteers' features!"
        return state
    
    
    async def chat(self, request: ChatRequest, user_id: str = None) -> ChatResponse:
        """Main chat interface with Redis-based conversation storage"""
        start_time = time.time()

        # Use session_id consistently
        session_id = request.session_id or f"session_{uuid.uuid4()}"
        thread_id = f"thread_{session_id}"

        # Check for existing workflow state
        config = {"configurable": {"thread_id": thread_id}}

        try:
            existing_state = await self.graph.aget_state(config)
            has_previous_state = existing_state and existing_state.values
        except Exception as e:
            logger.error(f"Failed to get existing state: {e}")
            has_previous_state = False

        initial_state = InferenceState(
            session_id=session_id,
            user_query=request.query,
            processed_query="",
            retrieved_docs=[],
            context="",
            response="",
            conversation_history=[],
            error="",
            stage="initialized",
            thread_id=thread_id,
            thread_ts=time.time()
        )

        # If we have previous state, carry forward conversation history
        if has_previous_state and existing_state.values:
            previous_values = existing_state.values
            if previous_values.get("conversation_history"):
                initial_state["conversation_history"] = previous_values["conversation_history"]

        try:
            # Execute with consistent thread_id for memory persistence
            final_state = await self.graph.ainvoke(initial_state, config=config)

            processing_time = time.time() - start_time

            return ChatResponse(
                response=final_state.get("response", ""),
                session_id=session_id,
                retrieved_docs=len(final_state.get("retrieved_docs", [])),
                context_used=len(final_state.get("context", "")) > 0,
                processing_time=processing_time,
                error=final_state.get("error", None)
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chat processing failed: {e}")

            return ChatResponse(
                response="I apologize, but I'm having trouble right now. Please try again or visit our platform to explore Impacteers' features!",
                session_id=session_id,
                retrieved_docs=0,
                context_used=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session from Redis"""
        try:
            history = await self.db_manager.get_conversation_history(session_id)
            return convert_all_types_to_serializable(history)
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

