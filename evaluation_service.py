"""
Evaluation service for the Impacteers RAG system
"""

import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime
import asyncio
import json
from sentence_transformers import SentenceTransformer

import time
import numpy as np
import uuid
import time
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import settings
from models import EvaluationResponse
from database import DatabaseManager
from inference_service import InferenceService
from embedding_service import SharedEmbeddingService

logger = logging.getLogger(__name__)


class EvaluationState(TypedDict):
    test_cases: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    generation_results: List[Dict[str, Any]]
    overall_metrics: Dict[str, float]
    evaluation_report: str
    error: str
    stage: str
    thread_id: str      # <-- ADD THIS
    thread_ts: float    # <-- OPTIONAL


class TestCaseGenerator:
    """Generate test cases for evaluation"""
    
    def __init__(self):
        self.test_cases = [
            {
                "query": "I'm looking for a job",
                "expected_answer": "Before exploring job opportunities, please sign up to get personalised job suggestions.",
                "category": "job_search",
                "difficulty": "easy",
                "relevant_categories": ["job_search_opportunities"]
            },
            {
                "query": "What courses do you offer?",
                "expected_answer": "We offer curated courses designed for career acceleration in diverse fields.",
                "category": "courses",
                "difficulty": "easy",
                "relevant_categories": ["courses_upskilling"]
            },
            {
                "query": "How can I assess my skills?",
                "expected_answer": "Take our free Skill Check by signing up — we'll guide you step-by-step.",
                "category": "assessment",
                "difficulty": "medium",
                "relevant_categories": ["skill_assessment"]
            },
            {
                "query": "I need mentorship for career change",
                "expected_answer": "We have mentors from various industries who can guide you through career transitions.",
                "category": "mentorship",
                "difficulty": "medium",
                "relevant_categories": ["mentorship"]
            },
            {
                "query": "What's IIPL and when does it happen?",
                "expected_answer": "IIPL is a sports and career development tournament that runs from August 5th to September 21st.",
                "category": "events",
                "difficulty": "hard",
                "relevant_categories": ["community_events"]
            },
            {
                "query": "Can you help me compare different career paths in tech vs design?",
                "expected_answer": "Based on your interests, we suggest exploring career clusters. A mentor can guide you deeper into comparing different paths.",
                "category": "career_guidance",
                "difficulty": "hard",
                "relevant_categories": ["mentorship", "platform_features"]
            }
        ]
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases"""
        return self.test_cases


class RetrievalEvaluator:
    """Evaluate retrieval performance"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_model):
        self.db_manager = db_manager
        self.embedding_model = embedding_model
    
    async def evaluate_retrieval(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate retrieval performance for test cases"""
        results = []
        
        for test_case in test_cases:
            query = test_case["query"]
            expected_categories = test_case["relevant_categories"]
            
            # Generate query embedding - handle both VertexAI and SentenceTransformer
            if hasattr(self.embedding_model, 'aembed_query'):
                # VertexAI embeddings (async)
                query_embedding = await self.embedding_model.aembed_query(query)
            else:
                # SentenceTransformer embeddings (sync, need to run in executor)
                import asyncio
                loop = asyncio.get_event_loop()
                query_embedding = await loop.run_in_executor(
                    None, self.embedding_model.encode, query
                )
                # Convert to list if it's a numpy array
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
            
            # Retrieve documents
            retrieved_docs = await self.db_manager.hybrid_search(query, query_embedding, top_k=5)
            
            # Evaluate retrieval
            metrics = self._calculate_retrieval_metrics(retrieved_docs, expected_categories)
            
            result = {
                "query": query,
                "category": test_case["category"],
                "difficulty": test_case["difficulty"],
                "retrieved_count": len(retrieved_docs),
                "relevant_retrieved": metrics["relevant_retrieved"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "avg_similarity": metrics["avg_similarity"]
            }
            
            results.append(result)
        
        return results
    


    def _calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Dict[str, Any]],
        expected_categories: List[str]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        if not retrieved_docs:
            return {
                "relevant_retrieved": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_similarity": 0.0
            }
        
        # Count relevant documents
        relevant_count = 0
        similarities = []
        
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            doc_category = metadata.get('category', '')
            similarity = doc.get('similarity', 0)
            similarities.append(similarity)
            
            if any(expected_cat in doc_category for expected_cat in expected_categories):
                relevant_count += 1
        
        # Calculate metrics
        precision = relevant_count / len(retrieved_docs)
        recall = relevant_count / len(expected_categories) if expected_categories else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    
        return {
            "relevant_retrieved": relevant_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_similarity": avg_similarity
        }

def safe_float(value):
    return value.item() if isinstance(value, np.generic) else float(value)
class GenerationEvaluator:
    """Evaluate response generation quality"""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
    
    async def evaluate_generation(self, test_cases: List[Dict[str, Any]], 
                                 generated_responses: List[str]) -> List[Dict[str, Any]]:
        """Evaluate generation quality"""
        results = []
        
        for test_case, generated_response in zip(test_cases, generated_responses):
            expected_answer = test_case["expected_answer"]
            
            # Evaluate response quality
            quality_metrics = await self._evaluate_response_quality(
                test_case["query"], 
                generated_response, 
                expected_answer
            )
            
            # Calculate similarity score
            similarity_score = self._calculate_semantic_similarity(
                generated_response, 
                expected_answer
            )
            
            result = {
                "query": test_case["query"],
                "category": test_case["category"],
                "difficulty": test_case["difficulty"],
                "generated_response": generated_response,
                "expected_answer": expected_answer,
                "relevance_score": float(quality_metrics["relevance"]),  # Convert to Python float
                "accuracy_score": float(quality_metrics["accuracy"]),    # Convert to Python float
                "helpfulness_score": float(quality_metrics["helpfulness"]),  # Convert to Python float
                "semantic_similarity": float(similarity_score),          # Convert to Python float
                "overall_score": float((quality_metrics["relevance"] + quality_metrics["accuracy"] + 
                                      quality_metrics["helpfulness"] + similarity_score) / 4)  # Convert to Python float
            }
            
            results.append(result)
        
        return results
    
    async def _evaluate_response_quality(self, query: str, response: str, expected: str) -> Dict[str, float]:
        """Evaluate response quality using LLM"""
        prompt = PromptTemplate(
            template="""
            Evaluate the quality of this AI response on a scale of 1-10:
            
            User Query: {query}
            AI Response: {response}
            Expected Answer: {expected}
            
            Rate the response on:
            1. Relevance (1-10): How well does it address the query?
            2. Accuracy (1-10): How accurate is the information?
            3. Helpfulness (1-10): How helpful is it to the user?
            
            Return ONLY a valid JSON object with numeric scores (no other text):
            {{"relevance": 8, "accuracy": 7, "helpfulness": 9}}
            """,
            input_variables=["query", "response", "expected"]
        )
        
        try:
            result = await self.llm.ainvoke(
                prompt.format(query=query, response=response, expected=expected)
            )
            
            # Clean the response content
            content = result.content.strip()
            
            # Try to extract JSON if there's extra text
            if not content.startswith('{'):
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{[^}]*\}', content)
                if json_match:
                    content = json_match.group()
                else:
                    # Fallback if no JSON found
                    logger.warning(f"No JSON found in LLM response: {content}")
                    return {"relevance": 0.5, "accuracy": 0.5, "helpfulness": 0.5}
            
            scores = json.loads(content)
            
            # Normalize scores to 0-1 range and ensure they're floats
            return {
                "relevance": float(scores.get("relevance", 5)) / 10,
                "accuracy": float(scores.get("accuracy", 5)) / 10,
                "helpfulness": float(scores.get("helpfulness", 5)) / 10
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {"relevance": 0.5, "accuracy": 0.5, "helpfulness": 0.5}
    
    def _calculate_semantic_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between responses"""
        # Simple word overlap-based similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class MetricsCalculator:
    """Calculate overall evaluation metrics"""

    def calculate_overall_metrics(self, retrieval_results: List[Dict], 
                                 generation_results: List[Dict]) -> Dict[str, float]:
        """Calculate overall system metrics"""
        # Retrieval metrics
        retrieval_metrics = {
            "avg_precision": safe_float(np.mean([r["precision"] for r in retrieval_results])),
            "avg_recall": safe_float(np.mean([r["recall"] for r in retrieval_results])),
            "avg_f1_score": safe_float(np.mean([r["f1_score"] for r in retrieval_results])),
            "avg_similarity": safe_float(np.mean([r["avg_similarity"] for r in retrieval_results])),
        }


        generation_metrics = {
            "avg_relevance": safe_float(np.mean([r["relevance_score"] for r in generation_results])),
            "avg_accuracy": safe_float(np.mean([r["accuracy_score"] for r in generation_results])),
            "avg_helpfulness": safe_float(np.mean([r["helpfulness_score"] for r in generation_results])),
            "avg_semantic_similarity": safe_float(np.mean([r["semantic_similarity"] for r in generation_results])),
            "avg_overall_score": safe_float(np.mean([r["overall_score"] for r in generation_results])),
        }

        system_score = safe_float((retrieval_metrics["avg_f1_score"] + generation_metrics["avg_overall_score"]) / 2)


        # Combined metrics
        overall_metrics = {
            **retrieval_metrics,
            **generation_metrics,
            "system_score": system_score
        }


        return overall_metrics


class ReportGenerator:
    """Generate evaluation reports"""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
    
    async def generate_report(self, overall_metrics: Dict[str, float], 
                            retrieval_results: List[Dict], 
                            generation_results: List[Dict]) -> str:
        """Generate comprehensive evaluation report"""
        
        # Prepare summary statistics
        retrieval_summary = self._summarize_retrieval_results(retrieval_results)
        generation_summary = self._summarize_generation_results(generation_results)
        recommendations = self._generate_recommendations(overall_metrics)
        
        # Generate report using LLM
        prompt = PromptTemplate(
            template="""
            Generate a comprehensive evaluation report for the RAG system:
            
            SYSTEM PERFORMANCE SUMMARY:
            - Overall System Score: {system_score:.3f}
            - Retrieval F1 Score: {retrieval_f1:.3f}
            - Generation Overall Score: {generation_score:.3f}
            
            RETRIEVAL PERFORMANCE:
            - Average Precision: {avg_precision:.3f}
            - Average Recall: {avg_recall:.3f}
            - Average Similarity: {avg_similarity:.3f}
            
            GENERATION PERFORMANCE:
            - Average Relevance: {avg_relevance:.3f}
            - Average Accuracy: {avg_accuracy:.3f}
            - Average Helpfulness: {avg_helpfulness:.3f}
            
            DETAILED ANALYSIS:
            
            Retrieval Analysis:
            {retrieval_summary}
            
            Generation Analysis:
            {generation_summary}
            
            RECOMMENDATIONS:
            {recommendations}
            
            Generate a detailed evaluation report with insights and actionable recommendations for system improvement.
            """,
            input_variables=["system_score", "retrieval_f1", "generation_score", 
                           "avg_precision", "avg_recall", "avg_similarity",
                           "avg_relevance", "avg_accuracy", "avg_helpfulness",
                           "retrieval_summary", "generation_summary", "recommendations"]
        )
        
        try:
            response = await self.llm.ainvoke(
                prompt.format(
                    system_score=overall_metrics["system_score"],
                    retrieval_f1=overall_metrics["avg_f1_score"],
                    generation_score=overall_metrics["avg_overall_score"],
                    avg_precision=overall_metrics["avg_precision"],
                    avg_recall=overall_metrics["avg_recall"],
                    avg_similarity=overall_metrics["avg_similarity"],
                    avg_relevance=overall_metrics["avg_relevance"],
                    avg_accuracy=overall_metrics["avg_accuracy"],
                    avg_helpfulness=overall_metrics["avg_helpfulness"],
                    retrieval_summary=retrieval_summary,
                    generation_summary=generation_summary,
                    recommendations=recommendations
                )
            )
            return response.content
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Evaluation Report\n\nSystem Score: {overall_metrics['system_score']:.3f}\nSee logs for detailed metrics."
    
    def _summarize_retrieval_results(self, results: List[Dict]) -> str:
        """Summarize retrieval results"""
        if not results:
            return "No retrieval results to summarize."
        
        category_performance = {}
        for result in results:
            category = result["category"]
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result["f1_score"])
        
        summary = "Retrieval Performance by Category:\n"
        for category, scores in category_performance.items():
            avg_score = np.mean(scores)
            summary += f"- {category}: {avg_score:.3f}\n"
        
        return summary
    
    def _summarize_generation_results(self, results: List[Dict]) -> str:
        """Summarize generation results"""
        if not results:
            return "No generation results to summarize."
        
        difficulty_performance = {}
        for result in results:
            difficulty = result["difficulty"]
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = []
            difficulty_performance[difficulty].append(result["overall_score"])
        
        summary = "Generation Performance by Difficulty:\n"
        for difficulty, scores in difficulty_performance.items():
            avg_score = np.mean(scores)
            summary += f"- {difficulty}: {avg_score:.3f}\n"
        
        return summary
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> str:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics["avg_precision"] < 0.7:
            recommendations.append("Improve document chunking and embedding quality")
        
        if metrics["avg_recall"] < 0.6:
            recommendations.append("Expand knowledge base coverage")
        
        if metrics["avg_relevance"] < 0.7:
            recommendations.append("Enhance query processing and context building")
        
        if metrics["avg_accuracy"] < 0.8:
            recommendations.append("Review and update knowledge base for accuracy")
        
        if metrics["system_score"] < 0.7:
            recommendations.append("Overall system performance needs improvement")
        
        if not recommendations:
            recommendations.append("System performing well, monitor for consistency")
        
        return "\n".join(f"- {rec}" for rec in recommendations)

def check_for_error(next_node: str):
    def _inner(state: EvaluationState) -> str:
        return "handle_error" if state.get("error") else next_node
    return _inner


class EvaluationService:
    """LangGraph-based evaluation service"""
    
    def __init__(self, db_manager: DatabaseManager, inference_service: InferenceService):
        self.db_manager = db_manager
        self.inference_service = inference_service
        self.llm = ChatVertexAI(
            model=settings.llm_model,
            project=settings.project_id,
            location=settings.location,
            temperature=settings.llm_temperature,
            model_kwargs={"convert_system_message_to_human": True},  # <-- FIXED
        )
        # self.embedding_model = VertexAIEmbeddings(
        #     model_name=settings.get_embedding_model(),
        #     project=settings.project_id,
        #     location=settings.location
        # )
        # self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-V2")
        self.embedding_model = SharedEmbeddingService.get_instance()

        
        # Initialize components
        self.test_generator = TestCaseGenerator()
        self.retrieval_evaluator = RetrievalEvaluator(self.db_manager, self.embedding_model)
        self.generation_evaluator = GenerationEvaluator(self.llm)
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator(self.llm)
        
        # Initialize memory checkpoint
        self.memory = MemorySaver()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(EvaluationState)

        # 1️⃣ Register every node in your pipeline
        workflow.add_node("prepare_test_cases", self._prepare_test_cases)
        workflow.add_node("evaluate_retrieval",   self._evaluate_retrieval)
        workflow.add_node("evaluate_generation",  self._evaluate_generation)
        workflow.add_node("calculate_metrics",    self._calculate_metrics)
        workflow.add_node("generate_report",      self._generate_report)
        workflow.add_node("handle_error",         self._handle_error)

        # 2️⃣ Set the starting node
        workflow.set_entry_point("prepare_test_cases")

        # 3️⃣ Chain them with error‑aware transitions
        workflow.add_conditional_edges("prepare_test_cases",
                                       check_for_error("evaluate_retrieval"))
        workflow.add_conditional_edges("evaluate_retrieval",
                                       check_for_error("evaluate_generation"))
        workflow.add_conditional_edges("evaluate_generation",
                                       check_for_error("calculate_metrics"))
        workflow.add_conditional_edges("calculate_metrics",
                                       check_for_error("generate_report"))
        workflow.add_conditional_edges("generate_report",
                                       check_for_error(END))

        # 4️⃣ If anything bubbles up to handle_error, end the graph
        workflow.add_edge("handle_error", END)

        # 5️⃣ Compile with your MemorySaver
        return workflow.compile(checkpointer=self.memory)

    async def _prepare_test_cases(self, state: EvaluationState) -> EvaluationState:
        """Prepare test cases for evaluation"""
        try:
            state["stage"] = "preparing_test_cases"
            test_cases = self.test_generator.get_test_cases()
            state["test_cases"] = test_cases
            return state
        except Exception as e:
            state["error"] = f"Test case preparation failed: {str(e)}"
            return state
    
    async def _evaluate_retrieval(self, state: EvaluationState) -> EvaluationState:
        """Evaluate retrieval performance"""
        try:
            state["stage"] = "evaluating_retrieval"
            retrieval_results = await self.retrieval_evaluator.evaluate_retrieval(state["test_cases"])
            state["retrieval_results"] = retrieval_results
            return state
        except Exception as e:
            state["error"] = f"Retrieval evaluation failed: {str(e)}"
            return state
    
    async def _evaluate_generation(self, state: EvaluationState) -> EvaluationState:
        """Evaluate generation performance"""
        try:
            state["stage"] = "evaluating_generation"
            
            # Generate responses using inference service
            generated_responses = []
            for test_case in state["test_cases"]:
                from models import ChatRequest
                request = ChatRequest(query=test_case["query"])
                response = await self.inference_service.chat(request)
                generated_responses.append(response.response)
            
            # Evaluate generation quality
            generation_results = await self.generation_evaluator.evaluate_generation(
                state["test_cases"], 
                generated_responses
            )
            state["generation_results"] = generation_results
            return state
        except Exception as e:
            state["error"] = f"Generation evaluation failed: {str(e)}"
            return state
    
    async def _calculate_metrics(self, state: EvaluationState) -> EvaluationState:
        """Calculate overall metrics"""
        try:
            state["stage"] = "calculating_metrics"
            overall_metrics = self.metrics_calculator.calculate_overall_metrics(
                state["retrieval_results"], 
                state["generation_results"]
            )
            state["overall_metrics"] = overall_metrics
            return state
        except Exception as e:
            state["error"] = f"Metrics calculation failed: {str(e)}"
            return state
    
    async def _generate_report(self, state: EvaluationState) -> EvaluationState:
        """Generate evaluation report"""
        try:
            state["stage"] = "generating_report"
            report = await self.report_generator.generate_report(
                state["overall_metrics"],
                state["retrieval_results"],
                state["generation_results"]
            )
            state["evaluation_report"] = report
            return state
        except Exception as e:
            state["error"] = f"Report generation failed: {str(e)}"
            return state
    
    async def _handle_error(self, state: EvaluationState) -> EvaluationState:
        """Handle pipeline errors"""
        logger.error(f"Pipeline error in stage {state.get('stage', 'unknown')}: {state.get('error', 'unknown')}")
        return state
    
    
    
    
    
    async def run_evaluation(self) -> EvaluationResponse:
        """Run the complete evaluation pipeline with consistent memory"""
        start_time = time.time()

        # FIXED: Use timestamp-based thread ID for evaluation runs
        evaluation_id = f"eval_{int(time.time())}"
        thread_id = f"evaluation_{evaluation_id}"  # Consistent for this evaluation run

        initial_state = EvaluationState(
            test_cases=[],
            retrieval_results=[],
            generation_results=[],
            overall_metrics={},
            evaluation_report="",
            error="",
            stage="initialized",
            thread_id=thread_id,      # Use consistent thread ID
            thread_ts=time.time()
        )

        # FIXED: Use consistent config for memory persistence
        config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)

            processing_time = time.time() - start_time

            # Save evaluation results with thread_id for tracking
            if final_state.get("overall_metrics"):
                evaluation_data = {
                    **final_state,
                    "evaluation_id": evaluation_id,
                    "thread_id": thread_id
                }
                await self.db_manager.save_evaluation(evaluation_data)

            return EvaluationResponse(
                success=not final_state.get("error"),
                overall_score=final_state.get("overall_metrics", {}).get("system_score", 0.0),
                retrieval_metrics={
                    k: v for k, v in final_state.get("overall_metrics", {}).items() 
                    if k.startswith("avg_") and k in ["avg_precision", "avg_recall", "avg_f1_score", "avg_similarity"]
                },
                generation_metrics={
                    k: v for k, v in final_state.get("overall_metrics", {}).items() 
                    if k.startswith("avg_") and k in ["avg_relevance", "avg_accuracy", "avg_helpfulness", "avg_overall_score"]
                },
                test_cases_count=len(final_state.get("test_cases", [])),
                evaluation_report=final_state.get("evaluation_report", ""),
                processing_time=processing_time,
                error=final_state.get("error"),
                evaluation_id=evaluation_id  # Include for tracking
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Evaluation failed: {e}")

            return EvaluationResponse(
                success=False,
                overall_score=0.0,
                retrieval_metrics={},
                generation_metrics={},
                test_cases_count=0,
                evaluation_report="",
                processing_time=processing_time,
                error=str(e),
                evaluation_id=evaluation_id
            )