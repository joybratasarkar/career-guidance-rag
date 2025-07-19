"""
Enhanced Inference service for the Impacteers RAG system with proper response templates
"""

import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime
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
    session_id: str
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


class ResponseTemplateManager:
    """Manage response templates for specific queries"""
    
    def __init__(self):
        self.templates = {
            # Job Search Templates
            "job_search_general": "Before exploring job opportunities, please sign up to get personalised job suggestions. You can browse open roles here: Jobs.",
            "job_search_latest": "Please create your Impacteers account to access curated job listings and opportunities handpicked for your profile. Visit: [jobs link]",
            "remote_jobs": "Discover a wide range of remote roles tailored to your skills. Signup to unlock personalized remote job recommendations. Check out: [jobs link]",
            "jobs_bangalore": "Sure! Here are jobs available in Bangalore: Jobs in Bangalore. Before applying, we recommend signing up for tailored recommendations.",
            "apple_jobs": "Great choice! Here's what we found related to Apple: Apple Jobs. Before applying, take a skill assessment to evaluate your strengths or check our courses to stand out.",
            "internships_freshers": "Yes! We've curated internships perfect for freshers. First, sign up to unlock personalised matches: Internships link",
            "interview_calls": "That can be frustrating! Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.",
            "tech_jobs": "Explore the latest tech roles on Impacteers. To enhance your chances, consider our tech skill certifications and get matched to leading employers: [courses link], [jobs link]",
            "final_year_jobs": "Yes, many companies offer pre-placement roles and fresher openings. [Sign up here](#) to get personalized job suggestions that align with your graduation timeline.",
            "trending_jobs": "Currently trending: data analysis, UI/UX, customer success, and full-stack roles. Sign up to explore more and get matched to hot openings.",
            "content_writing_jobs": "Roles like content strategist, copywriter, SEO writer, and blog manager could be a fit. Sign up to get content-specific job alerts.",
            "highest_paying_freshers": "Roles in sales engineering, product design, and data analytics offer great starting packages. Sign up to compare salaries and apply now.",
            "job_match_score": "Great question! We use an AI Job Match Score that instantly tells you how well your profile matches a job — based on your skills, experience, and role requirements. Just sign up to unlock your score and boost your hiring chances.",
            
            # Courses Templates
            "courses_general": "We offer curated courses designed for career acceleration in diverse fields. Explore personalized learning paths and unlock new opportunities: [courses link]",
            "data_science_courses": "We've curated top-rated data science courses from trusted sources. Explore them here: Data Science Courses. For a smarter start, take our skill check first so we can match you better.",
            "coding_courses": "Impacteers hosts beginner to advanced coding programs with mentor support and real-world projects. Get started here: [courses link]",
            "product_management_courses": "Awesome goal! Start by checking our recommended Product Management courses, and if you'd like, take our skill assessment to know where to begin.",
            "uiux_courses": "You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.",
            "free_courses": "Absolutely. Here's a list of top free courses: Free Courses. And if you'd like a customized plan, take a quick skill check.",
            "courses_free": "Many of our courses are free or low-cost to help you learn without limits. Check them out here: [courses link]. To begin, sign up and personalize your experience!",
            "data_analysis_courses": "Absolutely! We offer practical, career-aligned courses in data analysis. Start here: [courses link]. Sign up to track your progress and unlock full access!",
            "failed_exams": "Absolutely. Impacteers is built for people at all learning stages. Whether you passed or failed, your growth starts here. Browse beginner-friendly courses: [link]. Signing up helps us guide you better!",
            "short_courses": "Totally understand! We offer short-format, high-impact courses that fit into a busy schedule. Explore them here: [courses link] — just sign up to access the right ones for your needs!",
            
            # Skill Assessment Templates
            "skills_unknown": "No worries! Most people discover hidden strengths through our assessment — like communication, logic, or leadership traits. Take our free Skill Check by signing up — we'll guide you step-by-step.",
            "interview_readiness": "Let's find out! We check soft skills, role clarity, and test answers using our prep tools. Sign up here to take the Interview Readiness Quiz and get detailed feedback.",
            "skill_testing": "Yup! We assess problem-solving, communication, and domain-specific skills. It's quick and free — just sign up to get started and see where you stand.",
            "test_before_course": "Not required, but highly recommended! Our skill assessment helps you choose the best course for your current level. Try it here: [link] — and sign up to get started!",
            "skills_confident": "Even if you're confident, our test can help match you to roles or courses you might not have considered. It's free, quick, and surprisingly insightful. Try it here: [assessment link]. Sign-up is required.",
            "assessment_accuracy": "They're designed by career experts and educators to give you real insight into your strengths. It's not just for fun — it's a stepping stone to smarter choices. Try it here: [link], once you sign up.",
            
            # Mentorship Templates  
            "mentorship_experienced": "Yes! We've got mentors from Flipkart, Infosys, and early-stage startups who guide 1-on-1. Sign up here to see mentor profiles and request a session",
            "career_path_help": "Based on your interests, we suggest exploring career clusters (like Design, Tech, Biz). A mentor can guide you deeper. Please sign up to access our career path tool & mentorship sessions.",
            "mentorship_available": "Absolutely! From resume reviews to portfolio prep, our mentors are ready to help. Sign up to browse them and book a free 15-minute intro call.",
            "professional_communities": "You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.",
            "career_stuck": "You're not alone — and yes, that's exactly what our mentors are here for. Sign up to get paired with someone who understands your path: [mentors link].",
            "data_science_mentor": "That's a great move! We have experienced data science mentors who've successfully transitioned themselves. Sign up here to get connected and receive personalized guidance: [mentorship link].",
            "sales_to_ai": "Absolutely! Many of our mentors have pivoted into AI from non-tech backgrounds. Sign up and we'll connect you with someone who understands your journey: [mentorship link].",
            "tech_start": "You're not alone! We've got mentors who help with exactly that — figuring out your first step into tech. Sign up here and we'll match you with the right guide: [mentorship link].",
            "ai_mentor": "Yes! Our AI experts can help you understand what it takes and how to begin. Sign up now to chat with someone who's already in the field: [mentorship link].",
            
            # Community Events Templates
            "hackathons": "Yes! We regularly host exciting hackathons, quizzes, and other contests to help you learn and win. Sign up here to explore upcoming events: [events link].",
            "iipl_info": "IIPL (Impacteers International Premier League) is an intra-college sports and career development tournament designed to empower students across Tamil Nadu. IIPL is an initiative by Impacteers to revolutionize how students engage with both sports and career-building. The tournament runs from August 5th to September 21st and is designed for team spirit, leadership, and digital career readiness.",
            "sports_events": "Yes! Alongside learning events, we host college sports leagues and cultural fests like IIPL — a mix of fun, competition, and networking. Sign up to be part of it: [community/events link]",
            "student_connect": "Definitely! Our community is filled with ambitious students across India. Join discussions, collaborate on projects, or compete in quizzes together — just sign up here: [community link].",
            "weekly_challenges": "We host weekly challenges, quizzes, and fun mini-events to keep learning engaging. You can participate by signing up here: [events/challenges link].",
            "showcase_skills": "Yes! Whether it's through hackathons, IIPL, or leaderboards, you'll find many ways to shine. Sign up and jump in: [events link].",
            "resume_building": "Absolutely! Join our community projects, hackathons, and volunteer teams to build real-world experience. Sign up here to get started: [community link].",
            "networking": "Yes! Our Impacteers community is the perfect place to find peers, mentors, and collaborators. Sign up and say hi: [community link]."
        }
        
        # Query patterns for template matching
        self.query_patterns = {
            r"(?i).*looking for.*job.*": "job_search_general",
            r"(?i).*jobs.*available.*": "job_search_general", 
            r"(?i).*latest.*job.*opening.*": "job_search_latest",
            r"(?i).*remote.*job.*": "remote_jobs",
            r"(?i).*jobs.*bangalore.*": "jobs_bangalore",
            r"(?i).*work.*apple.*": "apple_jobs",
            r"(?i).*internship.*fresh.*": "internships_freshers",
            r"(?i).*not getting.*interview.*": "interview_calls",
            r"(?i).*tech.*job.*": "tech_jobs",
            r"(?i).*final year.*job.*": "final_year_jobs",
            r"(?i).*trending.*job.*": "trending_jobs",
            r"(?i).*content writing.*job.*": "content_writing_jobs",
            r"(?i).*highest paying.*fresh.*": "highest_paying_freshers",
            r"(?i).*good fit.*job.*": "job_match_score",
            
            r"(?i).*upskill.*": "courses_general",
            r"(?i).*courses.*offer.*": "courses_general",
            r"(?i).*data science.*course.*": "data_science_courses",
            r"(?i).*learn.*coding.*": "coding_courses",
            r"(?i).*product management.*": "product_management_courses",
            r"(?i).*ui.*ux.*": "uiux_courses",
            r"(?i).*free.*course.*": "free_courses",
            r"(?i).*courses.*free.*": "courses_free",
            r"(?i).*data analysis.*": "data_analysis_courses",
            r"(?i).*failed.*exam.*": "failed_exams",
            r"(?i).*short.*course.*": "short_courses",
            
            r"(?i).*don't know.*skill.*": "skills_unknown",
            r"(?i).*ready.*interview.*": "interview_readiness",
            r"(?i).*test.*skill.*": "skill_testing",
            r"(?i).*test.*before.*course.*": "test_before_course",
            r"(?i).*already know.*skill.*": "skills_confident",
            r"(?i).*assessment.*accurate.*": "assessment_accuracy",
            
            r"(?i).*help.*experienced.*": "mentorship_experienced",
            r"(?i).*career path.*": "career_path_help",
            r"(?i).*mentorship.*available.*": "mentorship_available",
            r"(?i).*professional.*communit.*": "professional_communities",
            r"(?i).*stuck.*career.*": "career_stuck",
            r"(?i).*data science.*mentor.*": "data_science_mentor",
            r"(?i).*sales.*ai.*": "sales_to_ai",
            r"(?i).*tech.*start.*": "tech_start",
            r"(?i).*ai.*mentor.*": "ai_mentor",
            
            r"(?i).*hackathon.*": "hackathons",
            r"(?i).*iipl.*": "iipl_info",
            r"(?i).*sport.*event.*": "sports_events",
            r"(?i).*connect.*student.*": "student_connect",
            r"(?i).*weekly.*challenge.*": "weekly_challenges",
            r"(?i).*showcase.*skill.*": "showcase_skills",
            r"(?i).*build.*resume.*": "resume_building",
            r"(?i).*network.*": "networking"
        }
    
    def get_template_response(self, query: str) -> str:
        """Get template response for query if pattern matches (excluding memory queries)"""
        # Skip template matching for memory-related queries
        memory_keywords = [
            "conversation history", "previous conversation", "what conversation", 
            "what did we discuss", "what have we talked", "our conversation"
        ]
        
        if any(keyword in query.lower() for keyword in memory_keywords):
            return ""  # Let memory handler take over
        
        # Continue with normal template matching
        for pattern, template_key in self.query_patterns.items():
            if re.match(pattern, query):
                return self.templates.get(template_key, "")
        return ""

class ResponseGenerator:
    """Generate responses using LLM with knowledge base priority and memory awareness"""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
        self.template_manager = ResponseTemplateManager()
        self.system_prompt = """You are an AI assistant for Impacteers, a career platform that helps students and professionals with job search, skills development, and career guidance.

Important Guidelines:
1. ALWAYS check conversation history first before responding
2. If user asks about previous conversations, conversation history, or "what did we discuss", ALWAYS reference the actual conversation history provided
3. Use specific information from context when available - PRIORITIZE CONTEXT OVER TEMPLATES
4. Always encourage users to sign up for personalized features
5. Provide specific, actionable advice with proper URLs from the context
6. If context doesn't contain relevant information, acknowledge this and provide general guidance
7. Keep responses concise but informative
8. Focus on Impacteers' features: jobs, courses, assessments, mentorship, and community
9. For specific queries about IIPL, mention it runs from August 5th to September 21st
10. For mentor queries, mention Flipkart, Infosys, and early-stage startups
11. For job fit queries, mention AI Job Match Score
12. ALWAYS use the actual URLs from the context when they are provided

MEMORY QUERIES HANDLING:
- If user asks about "conversation history", "what we discussed", "previous conversation", or similar:
  - ALWAYS reference the actual conversation history
  - Summarize what was actually discussed
  - Do NOT give generic responses about internships or tech jobs unless that's what was actually discussed

Context: {context}

Conversation History: {history}

User Question: {query}

Response:"""
    
    async def generate_response(self, query: str, context: str, history: List[Dict]) -> str:
        """Generate response with knowledge base priority and enhanced memory awareness"""
        
        # Check for memory-related queries FIRST
        memory_keywords = [
            "conversation history", "previous conversation", "what conversation", 
            "what did we discuss", "what have we talked", "our conversation",
            "conversation we have", "what we discussed", "chat history",
            "previous messages", "earlier conversation"
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
        
        # PRIORITY 2: Template fallback only if no good context available
        template_response = self.template_manager.get_template_response(query)
        if template_response:
            logger.info(f"Using template fallback for query: {query[:50]}...")
            return template_response
        
        # PRIORITY 3: Generic fallback
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

        # Error-handling condition function
        def check_for_error(state: InferenceState) -> str:
            return "handle_error" if state.get("error") else "continue"

        # Add conditional edges for each node with error handling
        workflow.add_conditional_edges(
            "load_history",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": "process_query"
            }
        )

        workflow.add_conditional_edges(
            "process_query",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": "retrieve_documents"
            }
        )

        workflow.add_conditional_edges(
            "retrieve_documents",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": "build_context"
            }
        )

        workflow.add_conditional_edges(
            "build_context",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": "generate_response"
            }
        )

        workflow.add_conditional_edges(
            "generate_response",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": "save_conversation"
            }
        )

        workflow.add_conditional_edges(
            "save_conversation",
            check_for_error,
            {
                "handle_error": "handle_error",
                "continue": END
            }
        )

        # Handle error ends the graph
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=self.memory)
    async def debug_database_conversations(self, session_id: str):
        """Debug method to check what's actually in the database"""
        try:
            # Check total conversations in database
            total_convs = await self.db_manager.conversations_collection.count_documents({})
            logger.info(f"Total conversations in database: {total_convs}")

            # Check conversations for this session
            session_convs = await self.db_manager.conversations_collection.count_documents({
                "session_id": session_id
            })
            logger.info(f"Conversations for session {session_id}: {session_convs}")

            # Get all conversations (limit 5 for debugging)
            all_convs = await self.db_manager.conversations_collection.find({}).limit(5).to_list(5)
            logger.info("Sample conversations in database:")
            for conv in all_convs:
                session = conv.get('session_id', 'N/A')
                query = conv.get('user_query', 'N/A')
                logger.info(f"  Session: {session}, Query: {query}")

            # Check for exact session match
            exact_match = await self.db_manager.conversations_collection.find({
                "session_id": session_id
            }).to_list(10)
            logger.info(f"Exact matches for {session_id}: {len(exact_match)}")

        except Exception as e:
            logger.error(f"Database debug failed: {e}")

    async def _load_history(self, state: InferenceState) -> InferenceState:
        """Load conversation history with enhanced debugging"""
        try:
            state["stage"] = "loading_history"
    
            logger.info(f"Loading history for session: {state['session_id']}")
            
            # DEBUG: Check database state first
            await self.debug_database_conversations(state['session_id'])
    
            # Load from database (long-term memory)
            db_history = await self.db_manager.get_conversation_history(
                state["session_id"], 
                limit=settings.max_conversation_history
            )
    
            logger.info(f"Database returned {len(db_history)} conversations")
    
            # Convert all types to serializable types
            serializable_history = convert_all_types_to_serializable(db_history)
            state["conversation_history"] = serializable_history
    
            # DEBUG: Log each conversation
            for i, conv in enumerate(serializable_history):
                user_query = conv.get('user_query', 'N/A')
                response_preview = conv.get('response', 'N/A')[:50] + "..." if len(conv.get('response', '')) > 50 else conv.get('response', 'N/A')
                logger.info(f"  Conversation {i+1}: '{user_query}' -> '{response_preview}'")
    
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
        """Save conversation to database with enhanced logging"""
        try:
            state["stage"] = "saving_conversation"

            # Convert all types to serializable before saving
            serializable_retrieved_docs = convert_all_types_to_serializable(state["retrieved_docs"])
            serializable_metadata = convert_all_types_to_serializable({"processed_query": state["processed_query"]})

            # DEBUG: Log what we're trying to save
            logger.info(f"Saving conversation for session {state['session_id']}")
            logger.info(f"User query: {state['user_query']}")
            logger.info(f"Response: {state['response'][:100]}...")

            # Save to database
            await self.db_manager.save_conversation(
                state["session_id"],
                state["user_query"],
                state["response"],
                serializable_retrieved_docs,
                serializable_metadata
            )

            logger.info(f"Successfully saved conversation to database")

            # UPDATE: Add current conversation to history for next request
            current_conversation = {
                "user_query": state["user_query"],
                "response": state["response"],
                "timestamp": datetime.utcnow().isoformat(),
                "retrieved_docs": serializable_retrieved_docs,
                "metadata": serializable_metadata
            }

            # Add to conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []

            state["conversation_history"].append(current_conversation)

            # Keep only recent conversations (last 10)
            state["conversation_history"] = state["conversation_history"][-10:]

            logger.info(f"Updated conversation history, now has {len(state['conversation_history'])} items")

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
    
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Main chat interface with enhanced memory debugging"""
        start_time = time.time()

        # Use consistent session-based thread mapping
        session_id = request.session_id or f"session_{uuid.uuid4()}"
        thread_id = f"thread_{session_id}"

        logger.info(f"Processing request for session: {session_id}, thread: {thread_id}")
        logger.info(f"User query: {request.query}")

        # Check for existing workflow state
        config = {"configurable": {"thread_id": thread_id}}

        try:
            existing_state = await self.graph.aget_state(config)
            if existing_state and existing_state.values:
                logger.info(f"Previous state exists with keys: {list(existing_state.values.keys())}")
                if 'conversation_history' in existing_state.values:
                    history_length = len(existing_state.values['conversation_history'])
                    logger.info(f"Previous state has {history_length} conversation history items")
            else:
                logger.info("No previous state found")
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
                logger.info(f"Carried forward {len(initial_state['conversation_history'])} conversations from previous state")

        try:
            # Execute with consistent thread_id for memory persistence
            final_state = await self.graph.ainvoke(initial_state, config=config)

            processing_time = time.time() - start_time

            # DEBUG: Log final state
            final_history_length = len(final_state.get('conversation_history', []))
            logger.info(f"Final conversation history length: {final_history_length}")

            if final_history_length > 0:
                logger.info("Final conversation history items:")
                for i, conv in enumerate(final_state.get('conversation_history', [])):
                    user_q = conv.get('user_query', 'N/A')
                    logger.info(f"  {i+1}. {user_q}")

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
        """Get conversation history for a session"""
        try:
            history = await self.db_manager.get_conversation_history(session_id)
            return convert_all_types_to_serializable(history)
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def test_template_matching(self):
        """Test method to verify template matching works correctly"""
        test_queries = [
            "I'm looking for a job",
            "Show me jobs in Bangalore", 
            "What's IIPL?",
            "Free courses available?",
            "Can I get help from someone experienced?",
            "Are there any hackathons?",
            "I don't know what skills I have"
        ]
        
        for query in test_queries:
            template_response = self.response_generator.template_manager.get_template_response(query)
            print(f"Query: {query}")
            print(f"Template Match: {'Yes' if template_response else 'No'}")
            if template_response:
                print(f"Response: {template_response[:100]}...")
            print("-" * 80)