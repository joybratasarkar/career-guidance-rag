"""
Simple Streamlit Chat Interface for Impacteers RAG System
"""

import streamlit as st
import requests
import uuid
import time
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="Impacteers Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = "http://localhost:6000"

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

def send_message(user_id: str, message: str) -> str:
    """Send message via REST API"""
    try:
        payload = {
            "query": message,
            "session_id": user_id
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def test_backend_connection():
    """Test if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

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
        
        # Backend URL configuration
        st.subheader("Configuration")
        st.text_input("Backend URL", value=BACKEND_URL, disabled=True)
    
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
                if backend_status:
                    response = send_message(st.session_state.user_id, prompt)
                    st.write(response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    error_msg = "Backend is not available. Please check the connection."
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
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