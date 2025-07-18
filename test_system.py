#!/usr/bin/env python3
"""
System testing for the Impacteers RAG system
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30


class RAGSystemTester:
    """Comprehensive system tester"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=TEST_TIMEOUT)
        self.session_id = "test_session_123"
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = await self.client.get("/health")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                print(f"✅ Health check passed: {result['response']['status']}")
            else:
                print(f"❌ Health check failed: {result['status_code']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_setup(self) -> Dict[str, Any]:
        """Test system setup"""
        print("🔧 Testing system setup...")
        
        try:
            response = await self.client.post("/setup")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                setup_data = result["response"]
                print(f"✅ Setup completed:")
                print(f"   Documents: {setup_data['ingestion']['documents_processed']}")
                print(f"   Chunks: {setup_data['ingestion']['chunks_created']}")
                print(f"   System score: {setup_data['evaluation']['overall_score']:.3f}")
            else:
                print(f"❌ Setup failed: {result['status_code']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_chat(self) -> Dict[str, Any]:
        """Test chat functionality"""
        print("💬 Testing chat functionality...")
        
        test_queries = [
            "I'm looking for a job",
            "What courses do you offer?",
            "How can I assess my skills?",
            "Tell me about mentorship",
            "What's IIPL?"
        ]
        
        results = []
        
        for query in test_queries:
            try:
                print(f"   Testing: '{query}'")
                
                start_time = time.time()
                response = await self.client.post(
                    "/chat",
                    json={"query": query, "session_id": self.session_id}
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "query": query,
                        "success": True,
                        "response": data["response"],
                        "retrieved_docs": data["retrieved_docs"],
                        "context_used": data["context_used"],
                        "response_time": response_time,
                        "processing_time": data["processing_time"]
                    }
                    print(f"   ✅ Response: {data['response'][:100]}...")
                    print(f"   📚 Retrieved: {data['retrieved_docs']} docs")
                    print(f"   ⏱️  Time: {response_time:.2f}s")
                else:
                    result = {
                        "query": query,
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "response_time": response_time
                    }
                    print(f"   ❌ Failed: HTTP {response.status_code}")
                
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["success"])
        print(f"💬 Chat test completed: {success_count}/{len(test_queries)} successful")
        
        return {
            "success": success_count == len(test_queries),
            "results": results,
            "success_rate": success_count / len(test_queries)
        }
    
    async def test_conversation_continuity(self) -> Dict[str, Any]:
        """Test conversation continuity"""
        print("🔄 Testing conversation continuity...")
        
        conversation_queries = [
            "I'm looking for a job",
            "What about remote jobs?",
            "How do I prepare for interviews?",
            "What skills should I focus on?"
        ]
        
        results = []
        
        for i, query in enumerate(conversation_queries):
            try:
                print(f"   [{i+1}] {query}")
                
                response = await self.client.post(
                    "/chat",
                    json={"query": query, "session_id": self.session_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "query": query,
                        "success": True,
                        "response": data["response"],
                        "retrieved_docs": data["retrieved_docs"]
                    }
                    print(f"   ✅ Response: {data['response'][:80]}...")
                else:
                    result = {
                        "query": query,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    print(f"   ❌ Failed: HTTP {response.status_code}")
                
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        # Test conversation history
        try:
            history_response = await self.client.get(f"/conversations/{self.session_id}")
            if history_response.status_code == 200:
                history = history_response.json()
                print(f"   📚 Conversation history: {len(history)} entries")
                continuity_success = len(history) >= len(conversation_queries)
            else:
                continuity_success = False
                print(f"   ❌ History fetch failed: HTTP {history_response.status_code}")
        except Exception as e:
            continuity_success = False
            print(f"   ❌ History error: {e}")
        
        success_count = sum(1 for r in results if r["success"])
        overall_success = success_count == len(conversation_queries) and continuity_success
        
        print(f"🔄 Continuity test: {success_count}/{len(conversation_queries)} successful")
        
        return {
            "success": overall_success,
            "results": results,
            "history_success": continuity_success
        }
    
    async def test_evaluation(self) -> Dict[str, Any]:
        """Test evaluation functionality"""
        print("📊 Testing evaluation...")
        
        try:
            response = await self.client.post("/evaluate")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
            
            if result["success"]:
                data = response.json()
                result.update({
                    "overall_score": data["overall_score"],
                    "test_cases": data["test_cases_count"],
                    "processing_time": data["processing_time"],
                    "retrieval_metrics": data["retrieval_metrics"],
                    "generation_metrics": data["generation_metrics"]
                })
                
                print(f"✅ Evaluation completed:")
                print(f"   Overall Score: {data['overall_score']:.3f}")
                print(f"   Test Cases: {data['test_cases_count']}")
                print(f"   Processing Time: {data['processing_time']:.2f}s")
            else:
                result["error"] = f"HTTP {response.status_code}"
                print(f"❌ Evaluation failed: HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_batch_chat(self) -> Dict[str, Any]:
        """Test batch chat functionality"""
        print("🧪 Testing batch chat...")
        
        try:
            response = await self.client.post("/test-chat")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
            
            if result["success"]:
                data = response.json()
                result["responses"] = data
                print(f"✅ Batch chat completed: {len(data)} responses")
                
                # Check response quality
                valid_responses = sum(1 for r in data if r["retrieved_docs"] > 0)
                result["quality_score"] = valid_responses / len(data)
                print(f"   Quality: {valid_responses}/{len(data)} with retrieved docs")
            else:
                result["error"] = f"HTTP {response.status_code}"
                print(f"❌ Batch chat failed: HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Batch chat error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_system_stats(self) -> Dict[str, Any]:
        """Test system statistics"""
        print("📈 Testing system statistics...")
        
        try:
            response = await self.client.get("/stats")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
            
            if result["success"]:
                data = response.json()
                result["stats"] = data
                print(f"✅ System stats:")
                print(f"   Documents: {data['documents_count']}")
                print(f"   Conversations: {data['conversations_count']}")
                print(f"   Latest eval score: {data['latest_evaluation']['score']:.3f}")
            else:
                result["error"] = f"HTTP {response.status_code}"
                print(f"❌ Stats failed: HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Running full test suite...")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Health check
        results["health"] = await self.test_health_check()
        
        # Test 2: Setup system
        if results["health"]["success"]:
            results["setup"] = await self.test_setup()
        else:
            print("⏭️  Skipping setup due to health check failure")
            results["setup"] = {"success": False, "skipped": True}
        
        # Test 3: Chat functionality
        if results["setup"]["success"]:
            results["chat"] = await self.test_chat()
        else:
            print("⏭️  Skipping chat due to setup failure")
            results["chat"] = {"success": False, "skipped": True}
        
        # Test 4: Conversation continuity
        if results["chat"]["success"]:
            results["continuity"] = await self.test_conversation_continuity()
        else:
            print("⏭️  Skipping continuity due to chat failure")
            results["continuity"] = {"success": False, "skipped": True}
        
        # Test 5: Evaluation
        if results["setup"]["success"]:
            results["evaluation"] = await self.test_evaluation()
        else:
            print("⏭️  Skipping evaluation due to setup failure")
            results["evaluation"] = {"success": False, "skipped": True}
        
        # Test 6: Batch chat
        if results["setup"]["success"]:
            results["batch_chat"] = await self.test_batch_chat()
        else:
            print("⏭️  Skipping batch chat due to setup failure")
            results["batch_chat"] = {"success": False, "skipped": True}
        
        # Test 7: System stats
        results["stats"] = await self.test_system_stats()
        
        # Calculate overall results
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r["success"])
        skipped_tests = sum(1 for r in results.values() if r.get("skipped", False))
        
        overall_success = successful_tests == total_tests
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result["success"] else "⏭️  SKIP" if result.get("skipped") else "❌ FAIL"
            print(f"{test_name.upper():<15} {status}")
        
        print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
        if skipped_tests > 0:
            print(f"Skipped: {skipped_tests} tests")
        
        success_rate = successful_tests / (total_tests - skipped_tests) if total_tests > skipped_tests else 0
        print(f"Success rate: {success_rate:.1%}")
        
        if overall_success:
            print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        else:
            print("\n⚠️  Some tests failed. Check logs for details.")
        
        return {
            "overall_success": overall_success,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "skipped_tests": skipped_tests,
            "results": results
        }


async def main():
    """Run the test suite"""
    print("🧪 Impacteers RAG System Test Suite")
    print("=" * 60)
    
    async with RAGSystemTester() as tester:
        results = await tester.run_full_test_suite()
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to test_results.json")
        
        # Exit with appropriate code
        exit(0 if results["overall_success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())