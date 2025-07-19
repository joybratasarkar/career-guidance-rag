"""
Complete Enhanced Ingestion service for the Impacteers RAG system with Web Content Integration - FIXED VERSION
"""

import logging
from typing import List, Dict, Any, TypedDict, Optional, Union
from datetime import datetime
import asyncio
import uuid
import time
import os
import re
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ML and NLP
import numpy as np
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_google_vertexai import ChatVertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    PyPDF2 = None

# Database
try:
    import motor.motor_asyncio
    from pymongo import MongoClient
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

# Configuration
@dataclass
class Settings:
    """Configuration settings"""
    # LLM settings
    llm_model: str = "gemini-1.5-pro"
    project_id: str = "your-gcp-project-id"
    location: str = "us-central1"
    llm_temperature: float = 0.1
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Database settings
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "impacteers_rag"
    collection_name: str = "documents"
    
    # Embedding settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

# Global settings instance
settings = Settings()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data Models
@dataclass
class DocumentInput:
    """Input document structure"""
    content: str
    document_type: str = "generic"  # pdf, manual, faq, web_content, etc.
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Processed document structure"""
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    doc_id: str
    created_at: str


class IngestionState(TypedDict):
    """State for the ingestion workflow"""
    documents: List[DocumentInput]
    web_urls: List[str]
    processed_chunks: List[Dict[str, Any]]
    stored_count: int
    error: str
    stage: str
    thread_id: str
    thread_ts: float


# Shared Embedding Service
class SharedEmbeddingService:
    """Singleton embedding service"""
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._model = SentenceTransformer(settings.embedding_model_name)
        return cls._model
    
    @classmethod
    def encode(cls, texts: List[str], **kwargs):
        model = cls.get_instance()
        return model.encode(texts, **kwargs)


# FIXED Database Manager
class DatabaseManager:
    """Enhanced database manager for document storage - FIXED VERSION"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.memory_storage = []
        self.use_mongodb = False  # Flag to track which storage we're using
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        if HAS_MONGODB:
            try:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
                self.db = self.client[settings.database_name]
                self.collection = self.db[settings.collection_name]
                self.use_mongodb = True
                logger.info("MongoDB connection initialized")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self._use_memory_storage()
        else:
            logger.warning("MongoDB not available, using in-memory storage")
            self._use_memory_storage()
    
    def _use_memory_storage(self):
        """Fallback to in-memory storage"""
        self.memory_storage = []
        self.collection = None
        self.use_mongodb = False
        logger.info("Using in-memory storage")
    
    async def store_documents(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store processed chunks in database - FIXED VERSION"""
        try:
            if self.use_mongodb and self.collection is not None:
                # MongoDB storage
                documents = []
                for chunk in chunks:
                    doc = {
                        "_id": chunk.get("doc_id", str(uuid.uuid4())),
                        "content": chunk["content"],
                        "embedding": chunk["embedding"],
                        "metadata": chunk["metadata"],
                        "created_at": chunk.get("created_at", datetime.utcnow().isoformat())
                    }
                    documents.append(doc)
                
                result = await self.collection.insert_many(documents, ordered=False)
                return [str(doc_id) for doc_id in result.inserted_ids]
            else:
                # Memory storage
                inserted_ids = []
                for chunk in chunks:
                    doc_id = chunk.get("doc_id", str(uuid.uuid4()))
                    doc = {
                        "_id": doc_id,
                        "content": chunk["content"],
                        "embedding": chunk["embedding"],
                        "metadata": chunk["metadata"],
                        "created_at": chunk.get("created_at", datetime.utcnow().isoformat())
                    }
                    self.memory_storage.append(doc)
                    inserted_ids.append(doc_id)
                
                return inserted_ids
                
        except Exception as e:
            logger.error(f"Failed to store documents: {e}")
            raise
    
    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if self.use_mongodb and self.collection is not None:
                # MongoDB vector search (simplified version)
                cursor = self.collection.find({}).limit(limit * 2)
                documents = await cursor.to_list(length=limit * 2)
                
                # Calculate cosine similarity
                similarities = []
                for doc in documents:
                    similarity = self._cosine_similarity(query_embedding, doc["embedding"])
                    similarities.append((similarity, doc))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in similarities[:limit]]
            
            else:
                # Memory storage search
                similarities = []
                for doc in self.memory_storage:
                    similarity = self._cosine_similarity(query_embedding, doc["embedding"])
                    similarities.append((similarity, doc))
                
                similarities.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in similarities[:limit]]
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if self.use_mongodb and self.collection is not None:
                count = await self.collection.count_documents({})
            else:
                count = len(self.memory_storage)
            
            return {
                "total_documents": count,
                "storage_type": "mongodb" if self.use_mongodb else "memory"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_documents": 0, "storage_type": "unknown"}


# Web Content Processor
class WebContentProcessor:
    """Enhanced web content processor for Impacteers URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # URL to category mapping
        self.url_category_map = {
            '/jobs': 'job_search_opportunities',
            '/courses': 'courses_upskilling', 
            '/skill-assessment': 'skill_assessment',
            '/career-assessment-test-ai': 'skill_assessment',
            '/mentor': 'mentorship',
            '/community': 'community_events',
            '/events': 'community_events',
            '/clubs': 'community_events',
            '/resume': 'platform_features',
            '/coverletter': 'platform_features',
            '/career-path': 'platform_features',
            '/learning-path': 'platform_features'
        }

    def get_category_from_url(self, url: str) -> str:
        """Determine category based on URL path"""
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        return self.url_category_map.get(path, 'platform_features')

    async def fetch_web_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch and process content from multiple URLs"""
        web_documents = []
        
        for url in urls:
            try:
                await asyncio.sleep(1)  # Rate limiting
                content_data = await self._fetch_single_url(url)
                if content_data:
                    web_documents.append(content_data)
                    logger.info(f"Successfully fetched content from {url}")
                else:
                    logger.warning(f"No content extracted from {url}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                continue
        
        return web_documents

    async def _fetch_single_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch content from a single URL"""
        try:
            # Make the request in a thread to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.session.get(url, timeout=10)
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} for {url}")
                return None
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract meaningful content
            extracted_content = self._extract_content(soup, url)
            
            if not extracted_content.strip():
                logger.warning(f"No meaningful content extracted from {url}")
                return None
            
            category = self.get_category_from_url(url)
            
            return {
                "content": extracted_content,
                "metadata": {
                    "source": "web",
                    "url": url,
                    "category": category,
                    "document_type": "web_page",
                    "fetch_timestamp": datetime.utcnow().isoformat(),
                    "title": self._extract_title(soup)
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract meaningful content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', '.main-content', '#main-content', '.content', 
            '#content', '.page-content', 'article', '.article-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        content_parts = []
        
        # Get page title
        title = self._extract_title(soup)
        if title:
            content_parts.append(f"Page Title: {title}")
        
        # Extract headings and paragraphs
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section']):
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short text
                text = re.sub(r'\s+', ' ', text)
                content_parts.append(text)
        
        # Extract list items
        for ul in main_content.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                text = li.get_text(strip=True)
                if text and len(text) > 10:
                    content_parts.append(f"• {text}")
        
        # Join and clean content
        full_content = '\n\n'.join(content_parts)
        full_content = re.sub(r'\n\s*\n\s*\n', '\n\n', full_content)
        
        return full_content.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return "Untitled Page"


# Enhanced Document Processor
class EnhancedDocumentProcessor:
    """Enhanced document processor that handles PDFs, web content, and existing knowledge"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
            length_function=len,
        )
        
        self.web_processor = WebContentProcessor()
        
        if HAS_PYMUPDF:
            logger.info("PyMuPDF (fitz) available for PDF processing")
        elif HAS_PYPDF2:
            logger.info("PyPDF2 available for PDF processing")
        else:
            logger.warning("No PDF processing libraries available - using sample content")

    async def process_mixed_content(self, documents: List[DocumentInput], web_urls: List[str] = None) -> List[Dict[str, Any]]:
        """Process documents and web content together"""
        processed_docs = []
        
        # Process traditional documents
        for doc_input in documents:
            try:
                if doc_input.document_type == "pdf":
                    processed = self._process_pdf_document(doc_input)
                elif doc_input.document_type == "manual":
                    processed = self._process_manual_pdf(doc_input)
                elif doc_input.document_type == "faq":
                    processed = self._process_faq_document(doc_input)
                elif doc_input.document_type == "web_content":
                    processed = self._process_web_document(doc_input)
                else:
                    processed = self._process_generic_document(doc_input)
                
                processed_docs.extend(processed)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_input.content[:100]}...: {e}")
                continue
        
        # Process web URLs
        if web_urls:
            try:
                web_documents = await self.web_processor.fetch_web_content(web_urls)
                processed_docs.extend(web_documents)
                logger.info(f"Successfully processed {len(web_documents)} web documents")
            except Exception as e:
                logger.error(f"Failed to process web URLs: {e}")
        
        return processed_docs

    def _process_pdf_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process PDF from file path"""
        try:
            if os.path.isfile(doc_input.content):
                text_content = self._extract_pdf_text(doc_input.content)
                filename = os.path.basename(doc_input.content)
            else:
                text_content = doc_input.content
                filename = doc_input.metadata.get("filename", "unknown.pdf")
            
            return [{
                "content": text_content,
                "metadata": {
                    **doc_input.metadata,
                    "category": doc_input.category,
                    "source": "pdf",
                    "filename": filename,
                    "document_type": "pdf"
                }
            }]
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return []

    def _process_manual_pdf(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process manual PDF content"""
        try:
            filename = doc_input.content
            
            if os.path.isfile(filename):
                text_content = self._extract_pdf_text(filename)
                logger.info(f"Extracted text from PDF file: {filename}")
            else:
                text_content = self._get_comprehensive_rag_content()
                logger.info(f"Using comprehensive RAG content for: {filename}")
            
            return [{
                "content": text_content,
                "metadata": {
                    **doc_input.metadata,
                    "category": doc_input.category,
                    "source": "manual_pdf",
                    "filename": filename,
                    "document_type": "manual"
                }
            }]
        except Exception as e:
            logger.error(f"Manual PDF processing failed: {e}")
            return []

    def _process_web_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process web document"""
        return [{
            "content": doc_input.content,
            "metadata": {
                **doc_input.metadata,
                "category": doc_input.category,
                "source": "web",
                "document_type": "web_content"
            }
        }]

    def _process_faq_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process FAQ document"""
        return [{
            "content": doc_input.content,
            "metadata": {
                **doc_input.metadata,
                "category": doc_input.category,
                "source": "faq",
                "document_type": "faq"
            }
        }]

    def _process_generic_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process generic document"""
        return [{
            "content": doc_input.content,
            "metadata": {
                **doc_input.metadata,
                "category": doc_input.category,
                "source": "generic",
                "document_type": "generic"
            }
        }]

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using available libraries"""
        try:
            text_content = ""
            
            if HAS_PYMUPDF:
                try:
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text_content += page.get_text()
                        text_content += "\n\n"
                    doc.close()
                    logger.info(f"Successfully extracted text using PyMuPDF from {pdf_path}")
                except Exception as e:
                    logger.error(f"PyMuPDF extraction failed: {e}")
                    text_content = ""
            
            if not text_content and HAS_PYPDF2:
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
            
            if not text_content:
                text_content = self._get_comprehensive_rag_content()
                logger.warning(f"Using fallback content for {pdf_path}")
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return self._get_comprehensive_rag_content()

    def _get_comprehensive_rag_content(self) -> str:
        """Get comprehensive RAG content based on the PDF scenarios"""
        return """
        RAG MODEL QUESTIONS - Complete Knowledge Base

        Category 1: Job Search & Opportunities

        User Query: I'm looking for a job. What jobs are available?
        Response: Before exploring job opportunities, please sign up to get personalised job suggestions. You can browse open roles here: Jobs.

        User Query: Show me the latest job openings.
        Response: Please create your Impacteers account to access curated job listings and opportunities handpicked for your profile. Visit: [jobs link]

        User Query: Any remote jobs?
        Response: Discover a wide range of remote roles tailored to your skills. Signup to unlock personalized remote job recommendations. Check out: [jobs link]

        User Query: Show me jobs in Bangalore
        Response: Sure! Here are jobs available in Bangalore: Jobs in Bangalore. Before applying, we recommend signing up for tailored recommendations.

        User Query: I want to work at Apple
        Response: Great choice! Here's what we found related to Apple: Apple Jobs. Before applying, take a skill assessment to evaluate your strengths or check our courses to stand out.

        User Query: Any internships for freshers?
        Response: Yes! We've curated internships perfect for freshers. First, sign up to unlock personalised matches: Internships link

        User Query: I'm not getting interview calls
        Response: That can be frustrating! Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.

        User Query: What tech jobs do you have?
        Response: Explore the latest tech roles on Impacteers. To enhance your chances, consider our tech skill certifications and get matched to leading employers: [courses link], [jobs link]

        User Query: I'm in my final year. Any jobs for me?
        Response: Yes, many companies offer pre-placement roles and fresher openings. [Sign up here](#) to get personalized job suggestions that align with your graduation timeline.

        User Query: Where are the trending jobs now?
        Response: Currently trending: data analysis, UI/UX, customer success, and full-stack roles. Sign up to explore more and get matched to hot openings.

        User Query: I'm good at content writing. What jobs suit me?
        Response: Roles like content strategist, copywriter, SEO writer, and blog manager could be a fit. Sign up to get content-specific job alerts.

        User Query: What's the highest paying job for freshers?
        Response: Roles in sales engineering, product design, and data analytics offer great starting packages. Sign up to compare salaries and apply now.

        User Query: How do I know if I'm a good fit for a job?
        Response: Great question! We use an AI Job Match Score that instantly tells you how well your profile matches a job — based on your skills, experience, and role requirements. Just sign up to unlock your score and boost your hiring chances.

        Category 2: Courses & Upskilling

        User Query: How can I upskill? What courses do you offer?
        Response: We offer curated courses designed for career acceleration in diverse fields. Explore personalized learning paths and unlock new opportunities: [courses link]

        User Query: Recommend courses for data science (or any domain).
        Response: We've curated top-rated data science courses from trusted sources. Explore them here: Data Science Courses. For a smarter start, take our skill check first so we can match you better.

        User Query: I want to learn coding.
        Response: Impacteers hosts beginner to advanced coding programs with mentor support and real-world projects. Get started here: [courses link]

        User Query: How do I upskill for a product management role?
        Response: Awesome goal! Start by checking our recommended Product Management courses, and if you'd like, take our skill assessment to know where to begin.

        User Query: I need to learn UI/UX
        Response: You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.

        User Query: Free courses available?
        Response: Absolutely. Here's a list of top free courses: Free Courses. And if you'd like a customized plan, take a quick skill check.

        User Query: Are the courses free?
        Response: Many of our courses are free or low-cost to help you learn without limits. Check them out here: [courses link]. To begin, sign up and personalize your experience!

        User Query: Can I learn data analysis here?
        Response: Absolutely! We offer practical, career-aligned courses in data analysis. Start here: [courses link]. Sign up to track your progress and unlock full access!

        User Query: I failed my college exams. Can I still learn here?
        Response: Absolutely. Impacteers is built for people at all learning stages. Whether you passed or failed, your growth starts here. Browse beginner-friendly courses: [link]. Signing up helps us guide you better!

        User Query: I don't have time for long courses. Do you have anything short and useful?
        Response: Totally understand! We offer short-format, high-impact courses that fit into a busy schedule. Explore them here: [courses link] — just sign up to access the right ones for your needs!

        Category 3: Skill Assessment

        User Query: I don't know what skills I have
        Response: No worries! Most people discover hidden strengths through our assessment — like communication, logic, or leadership traits. Take our free Skill Check by signing up — we'll guide you step-by-step.

        User Query: Am I ready for interviews?
        Response: Let's find out! We check soft skills, role clarity, and test answers using our prep tools. Sign up here to take the Interview Readiness Quiz and get detailed feedback.

        User Query: Can you test my skills?
        Response: Yup! We assess problem-solving, communication, and domain-specific skills. It's quick and free — just sign up to get started and see where you stand.

        User Query: Do I need to complete a test before taking a course?
        Response: Not required, but highly recommended! Our skill assessment helps you choose the best course for your current level. Try it here: [link] — and sign up to get started!

        User Query: I already know my skills. Why should I take this test?
        Response: Even if you're confident, our test can help match you to roles or courses you might not have considered. It's free, quick, and surprisingly insightful. Try it here: [assessment link]. Sign-up is required.

        User Query: Are these assessments scientifically accurate or just for fun?
        Response: They're designed by career experts and educators to give you real insight into your strengths. It's not just for fun — it's a stepping stone to smarter choices. Try it here: [link], once you sign up.

        Category 4: Mentorship

        User Query: Can I get help from someone experienced?
        Response: Yes! We've got mentors from Flipkart, Infosys, and early-stage startups who guide 1-on-1. Sign up here to see mentor profiles and request a session

        User Query: I need help choosing a career path
        Response: Based on your interests, we suggest exploring career clusters (like Design, Tech, Biz). A mentor can guide you deeper. Please sign up to access our career path tool & mentorship sessions.

        User Query: Is mentorship available here?
        Response: Absolutely! From resume reviews to portfolio prep, our mentors are ready to help. Sign up to browse them and book a free 15-minute intro call.

        User Query: Can I interact with other professionals? Any communities?
        Response: You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.

        User Query: I feel stuck in my career. Can a mentor help me?
        Response: You're not alone — and yes, that's exactly what our mentors are here for. Sign up to get paired with someone who understands your path: [mentors link].

        User Query: I want to talk to a professional in data science. I'm planning a career switch.
        Response: That's a great move! We have experienced data science mentors who've successfully transitioned themselves. Sign up here to get connected and receive personalized guidance: [mentorship link].

        User Query: Can someone guide me if I want to shift from sales to AI?
        Response: Absolutely! Many of our mentors have pivoted into AI from non-tech backgrounds. Sign up and we'll connect you with someone who understands your journey: [mentorship link].

        User Query: I'm lost. I want to get into tech but don't know where to start. Can I talk to someone?
        Response: You're not alone! We've got mentors who help with exactly that — figuring out your first step into tech. Sign up here and we'll match you with the right guide: [mentorship link].

        User Query: Is there a mentor who can help me understand if AI is right for me?
        Response: Yes! Our AI experts can help you understand what it takes and how to begin. Sign up now to chat with someone who's already in the field: [mentorship link].

        Category 5: Community & Events

        User Query: Are there any competitions or hackathons I can join?
        Response: Yes! We regularly host exciting hackathons, quizzes, and other contests to help you learn and win. Sign up here to explore upcoming events: [events link].

        User Query: What's IIPL? I saw it mentioned somewhere
        Response: IIPL (Impacteers International Premier League) is a intra-college sports and career development tournament designed to empower students across Tamil Nadu. IIPL is an initiative by Impacteers to revolutionize how students engage with both sports and career-building. The tournament runs from August 5th to September 21st and is designed for team spirit, leadership, and digital career readiness.

        User Query: Do you organize any sports or fun events for students?
        Response: Yes! Alongside learning events, we host college sports leagues and cultural fests like IIPL — a mix of fun, competition, and networking. Sign up to be part of it: [community/events link]

        User Query: Can I connect with students from other colleges?
        Response: Definitely! Our community is filled with ambitious students across India. Join discussions, collaborate on projects, or compete in quizzes together — just sign up here: [community link].

        User Query: How can I take part in your quizzes or weekly challenges?
        Response: We host weekly challenges, quizzes, and fun mini-events to keep learning engaging. You can participate by signing up here: [events/challenges link].

        User Query: Is there a way to showcase my skills or compete with others?
        Response: Yes! Whether it's through hackathons, IIPL, or leaderboards, you'll find many ways to shine. Sign up and jump in: [events link].

        User Query: I'm looking to build my resume. Are there any community projects or events I can join?
        Response: Absolutely! Join our community projects, hackathons, and volunteer teams to build real-world experience. Sign up here to get started: [community link].

        User Query: Is there a place where I can network or find like-minded learners?
        Response: Yes! Our Impacteers community is the perfect place to find peers, mentors, and collaborators. Sign up and say hi: [community link].
        """

    def create_enhanced_semantic_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create enhanced semantic chunks optimized for Q&A scenarios"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            source = doc["metadata"].get("source", "unknown")
            
            # Skip if content is too short
            if len(content.strip()) < 50:
                logger.warning(f"Document content too short, skipping: {content[:50]}...")
                continue
            
            # Different chunking strategies based on source
            if source == "web":
                semantic_chunks = self._chunk_web_content(content)
            elif "User Query" in content and "Response" in content:
                semantic_chunks = self._chunk_qa_content(content)
            else:
                semantic_chunks = self._identify_semantic_sections(content)
            
            for i, chunk_text in enumerate(semantic_chunks):
                if len(chunk_text.strip()) < 20:
                    continue
                    
                chunk = {
                    "content": chunk_text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "total_chunks": len(semantic_chunks),
                        "chunk_length": len(chunk_text.strip()),
                        "chunking_method": f"semantic_{source}"
                    }
                }
                chunks.append(chunk)
        
        return chunks

    def _chunk_qa_content(self, text: str) -> List[str]:
        """Special chunking for Q&A content"""
        chunks = []
        current_qa = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('User Query:'):
                # Save previous Q&A pair if exists
                if current_qa.strip() and len(current_qa.strip()) > 50:
                    chunks.append(current_qa.strip())
                
                # Start new Q&A pair
                current_qa = line + '\n'
            elif line.startswith('Response:'):
                current_qa += line + '\n'
            elif line and current_qa:
                current_qa += line + '\n'
        
        # Add the last Q&A pair
        if current_qa.strip() and len(current_qa.strip()) > 50:
            chunks.append(current_qa.strip())
        
        return chunks if chunks else self.text_splitter.split_text(text)

    def _chunk_web_content(self, text: str) -> List[str]:
        """Special chunking for web content"""
        return self.text_splitter.split_text(text)

    def _identify_semantic_sections(self, text: str) -> List[str]:
        """Enhanced semantic section identification"""
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_section.strip():
                    current_section += '\n'
                continue
            
            # Check if line is a header or major section boundary
            is_header = (
                line.endswith(':') or 
                (line.isupper() and len(line) > 3) or
                any(keyword in line.lower() for keyword in [
                    'category', 'user query', 'response', 'features', 'benefits', 
                    'how to', 'steps', 'guide', 'services', 'opportunities'
                ]) or
                line.startswith('Category') or
                re.match(r'^[A-Z][^.!?]*[.!?],', line)  # Sentence-like structure
            )
            if is_header:
                # Save previous section if it exists and is substantial
                if current_section.strip() and len(current_section.strip()) >= 100:
                    sections.append(current_section.strip())
                
                # Start new section with header
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add the last section
        if current_section.strip() and len(current_section.strip()) >= 100:
            sections.append(current_section.strip())
        
        # If no semantic sections found or sections are too few, fall back to regular chunking
        if len(sections) <= 1:
            return self.text_splitter.split_text(text)
        
        # Ensure sections are not too long
        final_sections = []
        for section in sections:
            if len(section) > settings.chunk_size * 1.5:
                # Split long sections further
                subsections = self.text_splitter.split_text(section)
                final_sections.extend(subsections)
            else:
                final_sections.append(section)
        
        return final_sections if final_sections else self.text_splitter.split_text(text)


# Enhanced Ingestion Service
class EnhancedIngestionService:
    """Enhanced ingestion service with web content integration"""
    
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
        self.processor = EnhancedDocumentProcessor()
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the enhanced ingestion workflow graph"""
        workflow = StateGraph(IngestionState)
        
        # Add nodes
        workflow.add_node("process_mixed_content", self._process_mixed_content)
        workflow.add_node("create_embeddings", self._create_embeddings)
        workflow.add_node("store_documents", self._store_documents)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("process_mixed_content")
        
        # Add edges
        workflow.add_conditional_edges(
            "process_mixed_content",
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

    async def _process_mixed_content(self, state: IngestionState) -> IngestionState:
        """Process documents and web content together"""
        try:
            state["stage"] = "processing_mixed_content"
            
            # Process both documents and web URLs
            processed_docs = await self.processor.process_mixed_content(
                state["documents"], 
                state.get("web_urls", [])
            )
            
            # Use enhanced semantic chunking
            state["processed_chunks"] = self.processor.create_enhanced_semantic_chunks(processed_docs)
            
            # Remove original documents to save memory
            state.pop("documents", None)
            state.pop("web_urls", None)
            
            logger.info(f"Processed {len(processed_docs)} documents into {len(state['processed_chunks'])} chunks")
            return state
            
        except Exception as e:
            logger.error(f"Mixed content processing failed: {e}")
            state["error"] = f"Mixed content processing failed: {e}"
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
            batch_size = settings.embedding_batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [chunk["content"] for chunk in batch]
                
                # Generate embeddings
                embeddings = SharedEmbeddingService.encode(
                    texts, 
                    show_progress_bar=True,
                    batch_size=min(batch_size, len(texts))
                )
                
                # Add embeddings and IDs to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    chunk["doc_id"] = f"doc_{uuid.uuid4()}"
                    chunk["created_at"] = datetime.utcnow().isoformat()
            
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

    async def ingest_with_web_content(
        self, 
        documents: List[DocumentInput], 
        web_urls: List[str] = None
    ) -> Dict[str, Any]:
        """Main ingestion method with web content integration"""
        start_time = time.time()
        
        # Create content-based thread ID for consistency
        content_items = [doc.content for doc in documents] + (web_urls or [])
        content_hash = hash(str(content_items))
        thread_id = f"enhanced_ingestion_{abs(content_hash)}"
        
        initial_state = IngestionState(
            documents=documents,
            web_urls=web_urls or [],
            processed_chunks=[],
            stored_count=0,
            error="",
            stage="initialized",
            thread_id=thread_id,
            thread_ts=time.time()
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            processing_time = time.time() - start_time
            
            return {
                "success": not final_state.get("error"),
                "error": final_state.get("error", ""),
                "documents_processed": len(documents),
                "web_urls_processed": len(web_urls) if web_urls else 0,
                "chunks_created": len(final_state.get("processed_chunks", [])),
                "stored_count": final_state.get("stored_count", 0),
                "processing_time": processing_time,
                "stage": final_state.get("stage", "unknown"),
                "thread_id": thread_id
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Enhanced ingestion failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "documents_processed": len(documents),
                "web_urls_processed": len(web_urls) if web_urls else 0,
                "chunks_created": 0,
                "stored_count": 0,
                "processing_time": processing_time,
                "stage": "failed",
                "thread_id": thread_id
            }

    async def get_comprehensive_sample_with_web(self) -> tuple[List[DocumentInput], List[str]]:
        """Get comprehensive sample including both documents and web URLs"""
        
        # Your Impacteers URLs
        web_urls = [
            "https://www.impacteers.com/courses",
            "https://www.impacteers.com/jobs", 
            "https://www.impacteers.com/mentor",
            "https://www.impacteers.com/learning-path",
            "https://www.impacteers.com/skill-assessment",
            "https://www.impacteers.com/career-path",
            "https://www.impacteers.com/resume",
            "https://www.impacteers.com/coverletter",
            "https://www.impacteers.com/career-assessment-test-ai",
            "https://www.impacteers.com/community",
            "https://www.impacteers.com/clubs",
            "https://www.impacteers.com/events"
        ]
        
        # Sample documents (including your PDFs)
        documents = [
            DocumentInput(
                content="rag_model_questions.pdf",  # Your PDF with Q&A scenarios
                document_type="manual", 
                category="job_search_opportunities", 
                metadata={"source": "knowledge_base", "filename": "rag_model_questions.pdf"}
            ),
            DocumentInput(
                content="platform_features.pdf",  # Any other PDF you have
                document_type="manual", 
                category="platform_features", 
                metadata={"source": "knowledge_base", "filename": "platform_features.pdf"}
            )
        ]
        
        return documents, web_urls

    # Backward compatibility methods
    async def ingest_documents(self, documents: List[DocumentInput]) -> Dict[str, Any]:
        """Backward compatibility - ingest documents without web content"""
        return await self.ingest_with_web_content(documents, [])

    async def get_sample_documents(self) -> List[DocumentInput]:
        """Backward compatibility - get sample documents only"""
        documents, _ = await self.get_comprehensive_sample_with_web()
        return documents


# Helper Classes
class ImpacteersRAGLoader:
    """Helper class to load all Impacteers content for RAG"""
    
    def __init__(self, ingestion_service: EnhancedIngestionService):
        self.ingestion_service = ingestion_service

    async def load_complete_knowledge_base(self) -> Dict[str, Any]:
        """Load complete Impacteers knowledge base including web content"""
        
        # Get both documents and web URLs
        documents, web_urls = await self.ingestion_service.get_comprehensive_sample_with_web()
        
        logger.info(f"Loading {len(documents)} documents and {len(web_urls)} web URLs")
        
        # Ingest everything together
        result = await self.ingestion_service.ingest_with_web_content(documents, web_urls)
        
        if result["success"]:
            logger.info(f"Successfully loaded complete knowledge base:")
            logger.info(f"- Documents processed: {result['documents_processed']}")
            logger.info(f"- Web URLs processed: {result['web_urls_processed']}")
            logger.info(f"- Total chunks created: {result['chunks_created']}")
            logger.info(f"- Chunks stored: {result['stored_count']}")
            logger.info(f"- Processing time: {result['processing_time']:.2f}s")
        else:
            logger.error(f"Failed to load knowledge base: {result['error']}")
        
        return result

    async def refresh_web_content_only(self) -> Dict[str, Any]:
        """Refresh only web content (useful for regular updates)"""
        
        _, web_urls = await self.ingestion_service.get_comprehensive_sample_with_web()
        
        # Create empty documents list, only process web URLs
        result = await self.ingestion_service.ingest_with_web_content([], web_urls)
        
        logger.info(f"Web content refresh result: {result}")
        return result

    async def load_custom_content(self, documents: List[DocumentInput], web_urls: List[str] = None) -> Dict[str, Any]:
        """Load custom combination of documents and web URLs"""
        return await self.ingestion_service.ingest_with_web_content(documents, web_urls or [])


# Query Service for RAG
class RAGQueryService:
    """Service for querying the RAG system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embedding_model = SharedEmbeddingService.get_instance()
        
    async def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the RAG system for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = SharedEmbeddingService.encode([query_text])[0]
            query_embedding = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            # Search for similar documents
            results = await self.db_manager.search_similar(query_embedding, limit=top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return await self.db_manager.get_stats()


# Main Application Class
class ImpacteersRAGSystem:
    """Complete Impacteers RAG System"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.ingestion_service = EnhancedIngestionService(self.db_manager)
        self.rag_loader = ImpacteersRAGLoader(self.ingestion_service)
        self.query_service = RAGQueryService(self.db_manager)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the RAG system with complete knowledge base"""
        logger.info("Initializing Impacteers RAG System...")
        
        result = await self.rag_loader.load_complete_knowledge_base()
        
        if result["success"]:
            logger.info("RAG System initialized successfully!")
        else:
            logger.error(f"RAG System initialization failed: {result['error']}")
        
        return result
    
    async def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the RAG system"""
        return await self.query_service.query(query_text, top_k)
    
    async def refresh_web_content(self) -> Dict[str, Any]:
        """Refresh web content"""
        return await self.rag_loader.refresh_web_content_only()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return await self.query_service.get_stats()


# Usage Examples and Main Function
async def example_usage():
    """Example of how to use the complete RAG system"""
    
    # Initialize the complete RAG system
    rag_system = ImpacteersRAGSystem()
    
    # Load the complete knowledge base
    print("Initializing RAG system...")
    init_result = await rag_system.initialize()
    print(f"Initialization result: {init_result}")
    
    if init_result["success"]:
        # Query the system
        print("\nQuerying the system...")
        query_results = await rag_system.query("I'm looking for data science jobs", top_k=3)
        
        print(f"Found {len(query_results)} relevant documents:")
        for i, result in enumerate(query_results, 1):
            print(f"\n{i}. {result['content'][:200]}...")
            print(f"   Category: {result['metadata'].get('category', 'unknown')}")
            print(f"   Source: {result['metadata'].get('source', 'unknown')}")
        
        # Get system stats
        stats = await rag_system.get_stats()
        print(f"\nSystem stats: {stats}")




# async def main():
#     """Main function to demonstrate the complete system"""
    
#     print("=== Impacteers Enhanced RAG System ===\n")
    
#     try:
#         # Run the example usage
#         await example_usage()
        
#         print("\n" + "="*50)
#         print("Custom ingestion example:")
#         await custom_ingestion_example()
        
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         print(f"Error: {e}")


# if __name__ == "__main__":
#     # Run the main function
#     asyncio.run(main())