"""
Enhanced Ingestion service for the Impacteers RAG system with complete knowledge base
"""

import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime
import asyncio
import uuid
import time
import os
from sentence_transformers import SentenceTransformer
from langchain_google_vertexai import ChatVertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF - better for text extraction
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
        if HAS_PYMUPDF:
            logger.info("PyMuPDF (fitz) available for PDF processing")
        elif HAS_PYPDF2:
            logger.info("PyPDF2 available for PDF processing")
        else:
            logger.warning("No PDF processing libraries available - using sample content")

    def process_documents(self, documents: List[DocumentInput]) -> List[Dict[str, Any]]:
        """Process documents based on their type"""
        processed_docs = []
        
        for doc_input in documents:
            try:
                if doc_input.document_type == "pdf":
                    processed = self._process_pdf_document(doc_input)
                elif doc_input.document_type == "manual":
                    # Handle manual PDF processing (from file path)
                    processed = self._process_manual_pdf(doc_input)
                elif doc_input.document_type == "faq":
                    processed = self._process_faq_document(doc_input)
                elif doc_input.document_type == "feature":
                    processed = self._process_feature_document(doc_input)
                else:
                    processed = self._process_generic_document(doc_input)
                
                processed_docs.extend(processed)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_input.content}: {e}")
                # Continue with other documents
                continue
        
        return processed_docs

    def _process_pdf_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process PDF from file path"""
        try:
            # Check if content is a file path
            if os.path.isfile(doc_input.content):
                text_content = self._extract_pdf_text(doc_input.content)
                filename = os.path.basename(doc_input.content)
            else:
                # If content is already text, use it directly
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
        """Process manual PDF - assume content is filename for now"""
        try:
            filename = doc_input.content
            
            # Check if it's an actual file path
            if os.path.isfile(filename):
                text_content = self._extract_pdf_text(filename)
                logger.info(f"Extracted text from PDF file: {filename}")
            else:
                # Use enhanced sample content based on filename and category
                text_content = self._get_enhanced_sample_content(filename, doc_input.category)
                logger.info(f"Using enhanced sample content for: {filename}")
            
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

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using available libraries"""
        try:
            text_content = ""
            
            if HAS_PYMUPDF:
                # Try PyMuPDF first (better text extraction)
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
            
            if not text_content and HAS_PYPDF2:
                # Fallback to PyPDF2
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
                # If both methods fail, return error message
                text_content = f"Could not extract text from {pdf_path}. Please install PyMuPDF or PyPDF2."
                logger.warning(f"No PDF processing libraries available for {pdf_path}")
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return f"Error extracting content from {pdf_path}: {e}"

    def _get_enhanced_sample_content(self, filename: str, category: str) -> str:
        """Get enhanced sample content based on filename and category"""
        
        # Handle your specific files
        if "feature.pdf" in filename.lower():
            return self._get_platform_features_content()
        elif "rag_knowledge.pdf" in filename.lower():
            # For rag_knowledge.pdf, provide comprehensive content covering all categories
            return self._get_comprehensive_rag_content()
        
        # Fallback to category-based content
        content_map = {
            "job_search_opportunities": self._get_job_search_content(),
            "courses_upskilling": self._get_courses_content(),
            "skill_assessment": self._get_skill_assessment_content(),
            "mentorship": self._get_mentorship_content(),
            "community_events": self._get_community_events_content(),
            "platform_features": self._get_platform_features_content()
        }
        
        # Return content based on category, or comprehensive content if category not found
        if category in content_map:
            return content_map[category]
        else:
            # Return all content combined
            return "\n\n".join(content_map.values())

    def _get_comprehensive_rag_content(self) -> str:
        """Comprehensive RAG knowledge content covering all categories"""
        return """
        Impacteers Comprehensive Knowledge Base - RAG Content
        
        JOB SEARCH AND OPPORTUNITIES:
        
        General Job Search:
        Before exploring job opportunities, please sign up to get personalised job suggestions.
        You can browse open roles and access curated job listings handpicked for your profile.
        
        Latest Job Openings:
        Please create your Impacteers account to access curated job listings and opportunities handpicked for your profile.
        
        Remote Job Opportunities:
        Discover a wide range of remote roles tailored to your skills. Signup to unlock personalized remote job recommendations.
        
        Location-Based Job Search:
        Show me jobs in Bangalore: Sure! Here are jobs available in Bangalore: Jobs in Bangalore. Before applying, we recommend signing up for tailored recommendations.
        
        Company-Specific Jobs:
        Apple Jobs: Great choice! Here's what we found related to Apple: Apple Jobs. Before applying, take a skill assessment to evaluate your strengths or check our courses to stand out.
        
        Internships for Freshers:
        Yes! We've curated internships perfect for freshers. First, sign up to unlock personalised matches.
        
        Tech Jobs and Opportunities:
        Explore the latest tech roles on Impacteers. To enhance your chances, consider our tech skill certifications and get matched to leading employers.
        
        Final Year Student Opportunities:
        Yes, many companies offer pre-placement roles and fresher openings. Sign up to get personalized job suggestions that align with your graduation timeline.
        
        Trending Jobs:
        Currently trending: data analysis, UI/UX, customer success, and full-stack roles. Sign up to explore more and get matched to hot openings.
        
        Content Writing Jobs:
        Roles like content strategist, copywriter, SEO writer, and blog manager could be a fit. Sign up to get content-specific job alerts.
        
        Highest Paying Jobs for Freshers:
        Roles in sales engineering, product design, and data analytics offer great starting packages. Sign up to compare salaries and apply now.
        
        AI Job Match Score:
        Great question! We use an AI Job Match Score that instantly tells you how well your profile matches a job — based on your skills, experience, and role requirements. Just sign up to unlock your score and boost your hiring chances.
        
        Interview Support:
        That can be frustrating! Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.
        
        COURSES AND UPSKILLING:
        
        Course Offerings:
        We offer curated courses designed for career acceleration in diverse fields. Explore personalized learning paths and unlock new opportunities.
        
        Data Science Courses:
        We've curated top-rated data science courses from trusted sources. Explore them here: Data Science Courses. For a smarter start, take our skill check first so we can match you better.
        
        Coding Programs:
        Impacteers hosts beginner to advanced coding programs with mentor support and real-world projects.
        
        Product Management Courses:
        Awesome goal! Start by checking our recommended Product Management courses, and if you'd like, take our skill assessment to know where to begin.
        
        UI/UX Design Courses:
        You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.
        
        Free Courses Available:
        Absolutely. Here's a list of top free courses: Free Courses. And if you'd like a customized plan, take a quick skill check.
        Many of our courses are free or low-cost to help you learn without limits. To begin, sign up and personalize your experience!
        
        Data Analysis Learning:
        Absolutely! We offer practical, career-aligned courses in data analysis. Sign up to track your progress and unlock full access!
        
        Learning for All Backgrounds:
        Absolutely. Impacteers is built for people at all learning stages. Whether you passed or failed, your growth starts here. Browse beginner-friendly courses. Signing up helps us guide you better!
        
        Short Format Courses:
        Totally understand! We offer short-format, high-impact courses that fit into a busy schedule. Explore them here — just sign up to access the right ones for your needs!
        
        SKILL ASSESSMENT AND TESTING:
        
        Skill Discovery:
        No worries! Most people discover hidden strengths through our assessment — like communication, logic, or leadership traits.
        Take our free Skill Check by signing up — we'll guide you step-by-step.
        
        Interview Readiness Assessment:
        Let's find out! We check soft skills, role clarity, and test answers using our prep tools.
        Sign up here to take the Interview Readiness Quiz and get detailed feedback.
        
        Comprehensive Skill Testing:
        Yup! We assess problem-solving, communication, and domain-specific skills. It's quick and free — just sign up to get started and see where you stand.
        
        Course Prerequisite Testing:
        Not required, but highly recommended! Our skill assessment helps you choose the best course for your current level. Try it here — and sign up to get started!
        
        Skills Validation:
        Even if you're confident, our test can help match you to roles or courses you might not have considered. It's free, quick, and surprisingly insightful. Try it here. Sign-up is required.
        
        Scientific Accuracy:
        They're designed by career experts and educators to give you real insight into your strengths. It's not just for fun — it's a stepping stone to smarter choices. Try it here, once you sign up.
        
        MENTORSHIP PROGRAM:
        
        Experienced Mentors:
        Yes! We've got mentors from Flipkart, Infosys, and early-stage startups who guide 1-on-1.
        Sign up here to see mentor profiles and request a session.
        
        Career Path Guidance:
        Based on your interests, we suggest exploring career clusters (like Design, Tech, Biz). A mentor can guide you deeper.
        Please sign up to access our career path tool & mentorship sessions.
        
        Mentorship Availability:
        Absolutely! From resume reviews to portfolio prep, our mentors are ready to help.
        Sign up to browse them and book a free 15-minute intro call.
        
        Career Transition Support:
        You're not alone — and yes, that's exactly what our mentors are here for. Sign up to get paired with someone who understands your path.
        
        Data Science Mentorship:
        That's a great move! We have experienced data science mentors who've successfully transitioned themselves. Sign up here to get connected and receive personalized guidance.
        
        Career Pivot Support:
        Absolutely! Many of our mentors have pivoted into AI from non-tech backgrounds. Sign up and we'll connect you with someone who understands your journey.
        
        Tech Career Guidance:
        You're not alone! We've got mentors who help with exactly that — figuring out your first step into tech. Sign up here and we'll match you with the right guide.
        
        AI Career Exploration:
        Yes! Our AI experts can help you understand what it takes and how to begin. Sign up now to chat with someone who's already in the field.
        
        COMMUNITY AND EVENTS:
        
        Competitions and Hackathons:
        Yes! We regularly host exciting hackathons, quizzes, and other contests to help you learn and win. Sign up here to explore upcoming events.
        
        IIPL Tournament Information:
        IIPL (Impacteers International Premier League) is an intra-college sports and career development tournament designed to empower students across Tamil Nadu. IIPL is an initiative by Impacteers to revolutionize how students engage with both sports and career-building.
        The tournament runs from August 5th to September 21st and is designed for team spirit, leadership, and digital career readiness.
        
        Sports and Fun Events:
        Yes! Alongside learning events, we host college sports leagues and cultural fests like IIPL — a mix of fun, competition, and networking. Sign up to be part of it.
        
        Student Networking:
        Definitely! Our community is filled with ambitious students across India. Join discussions, collaborate on projects, or compete in quizzes together — just sign up here.
        
        Weekly Challenges:
        We host weekly challenges, quizzes, and fun mini-events to keep learning engaging. You can participate by signing up here.
        
        Skill Showcase Opportunities:
        Yes! Whether it's through hackathons, IIPL, or leaderboards, you'll find many ways to shine. Sign up and jump in.
        
        Resume Building Events:
        Absolutely! Join our community projects, hackathons, and volunteer teams to build real-world experience. Sign up here to get started.
        
        Networking Platform:
        Yes! Our Impacteers community is the perfect place to find peers, mentors, and collaborators. Sign up and say hi.
        """

    def _get_job_search_content(self) -> str:
        """Complete job search and opportunities content"""
        return """
        Impacteers Job Search and Opportunities

        Job Search Features:
        Before exploring job opportunities, please sign up to get personalised job suggestions.
        You can browse open roles and access curated job listings handpicked for your profile.
        
        Location-Based Job Search:
        Show me jobs in Bangalore: Sure! Here are jobs available in Bangalore: Jobs in Bangalore. Before applying, we recommend signing up for tailored recommendations.
        
        Remote Job Opportunities:
        Discover a wide range of remote roles tailored to your skills. Signup to unlock personalized remote job recommendations.
        
        Company-Specific Jobs:
        Apple Jobs: Great choice! Here's what we found related to Apple: Apple Jobs. Before applying, take a skill assessment to evaluate your strengths or check our courses to stand out.
        
        Internships for Freshers:
        Yes! We've curated internships perfect for freshers. First, sign up to unlock personalised matches.
        
        Tech Jobs and Opportunities:
        Explore the latest tech roles on Impacteers. To enhance your chances, consider our tech skill certifications and get matched to leading employers.
        
        Final Year Student Opportunities:
        Yes, many companies offer pre-placement roles and fresher openings. Sign up to get personalized job suggestions that align with your graduation timeline.
        
        Trending Jobs:
        Currently trending: data analysis, UI/UX, customer success, and full-stack roles. Sign up to explore more and get matched to hot openings.
        
        Content Writing Jobs:
        Roles like content strategist, copywriter, SEO writer, and blog manager could be a fit. Sign up to get content-specific job alerts.
        
        Highest Paying Jobs for Freshers:
        Roles in sales engineering, product design, and data analytics offer great starting packages. Sign up to compare salaries and apply now.
        
        AI Job Match Score:
        Great question! We use an AI Job Match Score that instantly tells you how well your profile matches a job — based on your skills, experience, and role requirements. Just sign up to unlock your score and boost your hiring chances.
        
        Interview Support:
        That can be frustrating! Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.
        """

    def _get_courses_content(self) -> str:
        """Complete courses and upskilling content"""
        return """
        Impacteers Courses and Upskilling Programs

        Course Offerings:
        We offer curated courses designed for career acceleration in diverse fields. Explore personalized learning paths and unlock new opportunities.
        
        Data Science Courses:
        We've curated top-rated data science courses from trusted sources. Explore them here: Data Science Courses. For a smarter start, take our skill check first so we can match you better.
        
        Coding Programs:
        Impacteers hosts beginner to advanced coding programs with mentor support and real-world projects.
        
        Product Management Courses:
        Awesome goal! Start by checking our recommended Product Management courses, and if you'd like, take our skill assessment to know where to begin.
        
        UI/UX Design Courses:
        You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.
        
        Free Courses Available:
        Absolutely. Here's a list of top free courses: Free Courses. And if you'd like a customized plan, take a quick skill check.
        Many of our courses are free or low-cost to help you learn without limits. To begin, sign up and personalize your experience!
        
        Data Analysis Learning:
        Absolutely! We offer practical, career-aligned courses in data analysis. Sign up to track your progress and unlock full access!
        
        Learning for All Backgrounds:
        Absolutely. Impacteers is built for people at all learning stages. Whether you passed or failed, your growth starts here. Browse beginner-friendly courses. Signing up helps us guide you better!
        
        Short Format Courses:
        Totally understand! We offer short-format, high-impact courses that fit into a busy schedule. Explore them here — just sign up to access the right ones for your needs!
        """

    def _get_skill_assessment_content(self) -> str:
        """Complete skill assessment content"""
        return """
        Impacteers Skill Assessment and Testing

        Skill Discovery:
        No worries! Most people discover hidden strengths through our assessment — like communication, logic, or leadership traits.
        Take our free Skill Check by signing up — we'll guide you step-by-step.
        
        Interview Readiness Assessment:
        Let's find out! We check soft skills, role clarity, and test answers using our prep tools.
        Sign up here to take the Interview Readiness Quiz and get detailed feedback.
        
        Comprehensive Skill Testing:
        Yup! We assess problem-solving, communication, and domain-specific skills. It's quick and free — just sign up to get started and see where you stand.
        
        Course Prerequisite Testing:
        Not required, but highly recommended! Our skill assessment helps you choose the best course for your current level. Try it here — and sign up to get started!
        
        Skills Validation:
        Even if you're confident, our test can help match you to roles or courses you might not have considered. It's free, quick, and surprisingly insightful. Try it here. Sign-up is required.
        
        Scientific Accuracy:
        They're designed by career experts and educators to give you real insight into your strengths. It's not just for fun — it's a stepping stone to smarter choices. Try it here, once you sign up.
        
        Career Development Support:
        That can be frustrating! Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.
        """

    def _get_mentorship_content(self) -> str:
        """Complete mentorship content"""
        return """
        Impacteers Mentorship Program

        Experienced Mentors:
        Yes! We've got mentors from Flipkart, Infosys, and early-stage startups who guide 1-on-1.
        Sign up here to see mentor profiles and request a session.
        
        Career Path Guidance:
        Based on your interests, we suggest exploring career clusters (like Design, Tech, Biz). A mentor can guide you deeper.
        Please sign up to access our career path tool & mentorship sessions.
        
        Mentorship Availability:
        Absolutely! From resume reviews to portfolio prep, our mentors are ready to help.
        Sign up to browse them and book a free 15-minute intro call.
        
        Professional Communities:
        You're in the right place! Check out our beginner-friendly UI/UX courses. These come with mentorship and portfolio support too.
        
        Career Transition Support:
        You're not alone — and yes, that's exactly what our mentors are here for. Sign up to get paired with someone who understands your path.
        
        Data Science Mentorship:
        That's a great move! We have experienced data science mentors who've successfully transitioned themselves. Sign up here to get connected and receive personalized guidance.
        
        Career Pivot Support:
        Absolutely! Many of our mentors have pivoted into AI from non-tech backgrounds. Sign up and we'll connect you with someone who understands your journey.
        
        Tech Career Guidance:
        You're not alone! We've got mentors who help with exactly that — figuring out your first step into tech. Sign up here and we'll match you with the right guide.
        
        AI Career Exploration:
        Yes! Our AI experts can help you understand what it takes and how to begin. Sign up now to chat with someone who's already in the field.
        """

    def _get_community_events_content(self) -> str:
        """Complete community events content"""
        return """
        Impacteers Community and Events

        Competitions and Hackathons:
        Yes! We regularly host exciting hackathons, quizzes, and other contests to help you learn and win. Sign up here to explore upcoming events.
        
        IIPL Tournament Information:
        IIPL (Impacteers International Premier League) is an intra-college sports and career development tournament designed to empower students across Tamil Nadu. IIPL is an initiative by Impacteers to revolutionize how students engage with both sports and career-building.
        The tournament runs from August 5th to September 21st and is designed for team spirit, leadership, and digital career readiness.
        
        Sports and Fun Events:
        Yes! Alongside learning events, we host college sports leagues and cultural fests like IIPL — a mix of fun, competition, and networking. Sign up to be part of it.
        
        Student Networking:
        Definitely! Our community is filled with ambitious students across India. Join discussions, collaborate on projects, or compete in quizzes together — just sign up here.
        
        Weekly Challenges:
        We host weekly challenges, quizzes, and fun mini-events to keep learning engaging. You can participate by signing up here.
        
        Skill Showcase Opportunities:
        Yes! Whether it's through hackathons, IIPL, or leaderboards, you'll find many ways to shine. Sign up and jump in.
        
        Resume Building Events:
        Absolutely! Join our community projects, hackathons, and volunteer teams to build real-world experience. Sign up here to get started.
        
        Networking Platform:
        Yes! Our Impacteers community is the perfect place to find peers, mentors, and collaborators. Sign up and say hi.
        """

    def _get_platform_features_content(self) -> str:
        """Complete platform features content"""
        return """
        Impacteers Platform Features and Services
        
        AI Job Match Score:
        Our AI Job Match Score instantly tells you how well your profile matches a job based on your skills, experience, and role requirements. Just sign up to unlock your score and boost your hiring chances.
        
        Interview Readiness Tools:
        Take our Interview Readiness Quiz to check soft skills, role clarity, and test answers using our prep tools. Sign up here to get detailed feedback.
        
        Resume Builder:
        Have you tried our resume builder and career assessments to strengthen your profile? Let's work on that together.
        
        Skill Assessment Platform:
        Take our comprehensive skill assessment to identify your strengths and areas for improvement. Get personalized recommendations for courses and learning paths based on your results.
        
        Mentorship Network:
        Connect with experienced professionals from Flipkart, Infosys, and early-stage startups who can guide your career journey. Get personalized advice, feedback, and support from mentors in your field.
        
        Course Platform:
        Access a wide range of courses designed to accelerate your career growth. Learn from industry experts and gain certifications that matter to employers. Choose from technical skills, soft skills, and leadership development programs.
        
        Community Features:
        Participate in networking events, workshops, and career fairs. Connect with like-minded professionals and expand your network. Join industry-specific groups and attend virtual and in-person events.
        
        Career Planning Tools:
        Work with career counselors to create a personalized career plan. Set goals, track progress, and adjust your strategy as needed. Explore different career paths and understand the requirements for each.
        
        Job Search Platform:
        Discover curated job openings from top companies across various industries. Our AI-powered matching system connects you with roles that align with your skills and career goals. Filter by location, salary, experience level, and industry to find your perfect match.
        """

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

    def _process_feature_document(self, doc_input: DocumentInput) -> List[Dict[str, Any]]:
        """Process feature document"""
        return [{
            "content": doc_input.content,
            "metadata": {
                **doc_input.metadata,
                "category": doc_input.category,
                "source": "feature",
                "document_type": "feature"
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

    def create_semantic_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create semantic chunks based on content structure"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            
            # Skip if content is too short
            if len(content.strip()) < 50:
                logger.warning(f"Document content too short, skipping: {content[:50]}...")
                continue
            
            # Try to identify semantic boundaries
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
                        "chunking_method": "semantic"
                    }
                }
                chunks.append(chunk)
        
        return chunks

    def _identify_semantic_sections(self, text: str) -> List[str]:
        """Identify semantic sections in text"""
        # Enhanced approach: split by meaningful boundaries
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
            if (line.endswith(':') or 
                (line.isupper() and len(line) > 3) or
                any(keyword in line.lower() for keyword in [
                    'features', 'benefits', 'how to', 'steps', 'guide', 
                    'services', 'opportunities', 'iipl', 'august', 'september',
                    'flipkart', 'infosys', 'ai job match', 'interview readiness'
                ]) or
                line.startswith('Category') or
                line.startswith('User Query') or
                line.startswith('Response')):
                
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
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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
            state["processed_chunks"] = self.processor.create_semantic_chunks(processed_docs)
            
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

    async def ingest_documents(self, documents: List[DocumentInput]) -> Dict[str, Any]:
        """Main ingestion method"""
        start_time = time.time()
        
        initial_state = IngestionState(
            documents=documents,
            processed_chunks=[],
            stored_count=0,
            error="",
            stage="initialized",
            thread_id=str(uuid.uuid4()),
            thread_ts=time.time()
        )
        
        try:
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state["thread_id"]}}
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": not final_state.get("error"),
                "error": final_state.get("error", ""),
                "documents_processed": len(documents),
                "chunks_created": len(final_state.get("processed_chunks", [])),
                "stored_count": final_state.get("stored_count", 0),
                "processing_time": processing_time,
                "stage": final_state.get("stage", "unknown")
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
                "stage": "failed"
            }

    async def get_comprehensive_sample_documents(self) -> List[DocumentInput]:
        """Get comprehensive sample documents covering all RAG model scenarios"""
        return [
            DocumentInput(
                content="feature.pdf", 
                document_type="manual", 
                category="platform_features", 
                metadata={"source": "knowledge_base", "filename": "feature.pdf"}
            ),
            DocumentInput(
                content="rag_knowledge.pdf", 
                document_type="manual", 
                category="job_search_opportunities", 
                metadata={"source": "knowledge_base", "filename": "rag_knowledge.pdf"}
            )
        ]

    async def get_sample_documents(self) -> List[DocumentInput]:
        """Get sample documents for testing (backward compatibility)"""
        return await self.get_comprehensive_sample_documents()