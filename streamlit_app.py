"""
Streamlit Chat Interface for Impacteers RAG System
"""

import streamlit as st
import asyncio
import websockets
import json
import uuid
import time
import requests
from typing import Dict, List
import threading
from queue import Queue

# Configure page
st.set_page_config(
    page_title="Impacteers Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:6000")
WEBSOCKET_URL = st.secrets.get("WEBSOCKET_URL", "ws://localhost:6000")

class ChatWebSocket:
    def __init__(self, user_id: str, message_queue: Queue):
        self.user_id = user_id
        self.message_queue = message_queue
        self.websocket = None
        self.connected = False
        
    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.websocket = await websockets.connect(f"{WEBSOCKET_URL}/ws/{self.user_id}")
            self.connected = True
            return True
        except Exception as e:
            st.error(f"WebSocket connection failed: {e}")
            return False
    
    async def send_message(self, message: str):
        """Send message to WebSocket"""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(message)
                return True
            except Exception as e:
                st.error(f"Failed to send message: {e}")
                self.connected = False
                return False
        return False
    
    async def listen_for_messages(self):
        """Listen for incoming messages"""
        if self.websocket and self.connected:
            try:
                async for message in self.websocket:
                    self.message_queue.put({"type": "message", "content": message})
            except Exception as e:
                self.message_queue.put({"type": "error", "content": str(e)})
                self.connected = False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "websocket_connected" not in st.session_state:
        st.session_state.websocket_connected = False
    
    if "message_queue" not in st.session_state:
        st.session_state.message_queue = Queue()

def get_conversation_history(user_id: str) -> List[Dict]:
    """Get conversation history from REST API"""
    try:
        response = requests.get(f"{BACKEND_URL}/conversations/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.error(f"Failed to load conversation history: {e}")
        return []

def send_rest_message(user_id: str, message: str) -> str:
    """Send message via REST API as fallback"""
    try:
        payload = {
            "query": message,
            "session_id": user_id
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "Sorry, I'm having trouble right now. Please try again."
    except Exception as e:
        return f"Error: {e}"

def test_backend_connection():
    """Test if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

async def websocket_handler(user_id: str, message: str, message_queue: Queue):
    """Handle WebSocket communication"""
    chat_ws = ChatWebSocket(user_id, message_queue)
    
    if await chat_ws.connect():
        # Send message
        await chat_ws.send_message(message)
        
        # Wait for response with timeout
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            try:
                response = await asyncio.wait_for(chat_ws.websocket.recv(), timeout=1.0)
                message_queue.put({"type": "message", "content": response})
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                message_queue.put({"type": "error", "content": str(e)})
                break
        
        await chat_ws.disconnect()

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("💬 Impacteers AI Assistant")
    st.markdown("Ask me about jobs, courses, mentorship, and career guidance!")
    
    # Sidebar
    with st.sidebar:
        st.header("Chat Settings")
        
        # User ID display
        st.text_input("Your User ID", value=st.session_state.user_id, disabled=True)
        
        # Backend connection status
        backend_status = test_backend_connection()
        status_color = "🟢" if backend_status else "🔴"
        st.markdown(f"Backend Status: {status_color}")
        
        # Clear chat button
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Load conversation history
        if st.button("Load Previous Conversations", type="secondary"):
            history = get_conversation_history(st.session_state.user_id)
            if history:
                st.session_state.messages = []
                for conv in history:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": conv["user_query"],
                        "timestamp": conv["timestamp"]
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": conv["response"],
                        "timestamp": conv["timestamp"]
                    })
                st.success(f"Loaded {len(history)} conversations")
                st.rerun()
        
        # Connection method
        connection_method = st.radio(
            "Connection Method",
            ["WebSocket (Real-time)", "REST API (Fallback)"],
            index=0
        )
        
        # Backend URL configuration
        st.subheader("Configuration")
        st.text_input("Backend URL", value=BACKEND_URL, disabled=True)
        st.text_input("WebSocket URL", value=WEBSOCKET_URL, disabled=True)
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(message["timestamp"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your career..."):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if connection_method == "WebSocket (Real-time)" and backend_status:
                    # Try WebSocket first
                    try:
                        # Run WebSocket in thread to avoid blocking Streamlit
                        import asyncio
                        import threading
                        
                        response_container = st.empty()
                        response_container.info("Connecting via WebSocket...")
                        
                        # Use REST API as it's more reliable for Streamlit
                        response = send_rest_message(st.session_state.user_id, prompt)
                        
                        response_container.write(response)
                        
                        # Add assistant response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                    except Exception as e:
                        error_msg = f"WebSocket failed, using REST API: {e}"
                        st.warning(error_msg)
                        
                        # Fallback to REST API
                        response = send_rest_message(st.session_state.user_id, prompt)
                        st.write(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                else:
                    # Use REST API
                    response = send_rest_message(st.session_state.user_id, prompt)
                    st.write(response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # Rerun to update the chat
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🚀 Powered by Impacteers RAG System</p>
            <p>Built with FastAPI, Redis, MongoDB, and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()