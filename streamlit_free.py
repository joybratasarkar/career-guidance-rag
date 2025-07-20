import streamlit as st
import requests
import json
import time
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="🎯 Impacteers AI Chat",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Impacteers AI Career Assistant</h1>
    <p>Your intelligent guide to jobs, courses, mentorship, and career growth!</p>
</div>
""", unsafe_allow_html=True)

# Simple in-memory chat (since we can't use Redis on free Streamlit)
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.user_id = f"user_{int(time.time())}"

# Template responses for fast replies (since we can't use full backend)
QUICK_RESPONSES = {
    "hello": "👋 Hi there! I'm your Impacteers AI assistant. I can help you with:\n\n• 💼 **Job Search**: Find opportunities at https://www.impacteers.com/jobs\n• 📚 **Courses**: Explore learning at https://www.impacteers.com/courses\n• 🎯 **Skill Assessment**: Test your skills at https://www.impacteers.com/assessments\n• 👥 **Mentorship**: Connect with experts at https://www.impacteers.com/mentorship\n\nWhat would you like to explore today?",
    
    "job": "💼 **Great! Let's find you the perfect job:**\n\n🔍 **Browse All Jobs**: https://www.impacteers.com/jobs\n\n📍 **Popular Categories:**\n• Data Science & Analytics\n• Product Management  \n• Marketing & Growth\n• Engineering & Tech\n• Consulting & Strategy\n\n💡 **Pro Tip**: Sign up at https://www.impacteers.com/signup to get personalized job recommendations!\n\nWhat type of role interests you most?",
    
    "course": "📚 **Awesome! Here are our top learning opportunities:**\n\n🎓 **Explore Courses**: https://www.impacteers.com/courses\n\n🔥 **Popular Categories:**\n• Product Management Fundamentals\n• Data Science Bootcamp\n• Digital Marketing Mastery\n• Leadership & Strategy\n• Technical Skills Development\n\n✨ **Interactive Learning**: Get hands-on experience with real projects!\n\nWhich skill would you like to develop?",
    
    "mentorship": "👥 **Perfect! Let's connect you with industry experts:**\n\n🌟 **Find Mentors**: https://www.impacteers.com/mentorship\n\n💼 **Expertise Areas:**\n• Career Transition Guidance\n• Industry-Specific Advice\n• Skill Development\n• Leadership Coaching\n• Startup & Entrepreneurship\n\n🎯 **1-on-1 Sessions**: Get personalized guidance from professionals who've been where you want to go!\n\nWhat area would you like mentorship in?",
    
    "assessment": "🎯 **Let's discover your strengths:**\n\n📊 **Take Assessments**: https://www.impacteers.com/assessments\n\n🧠 **Available Tests:**\n• Personality & Work Style\n• Technical Skills Evaluation\n• Leadership Potential\n• Career Aptitude\n• Interview Preparation\n\n💡 **Get Insights**: Understand your strengths and areas for growth!\n\nWhich assessment interests you?",
}

def get_ai_response(user_input: str) -> str:
    """Generate AI response using template matching"""
    user_lower = user_input.lower()
    
    # Quick template matching
    if any(word in user_lower for word in ["hi", "hello", "hey", "start"]):
        return QUICK_RESPONSES["hello"]
    elif any(word in user_lower for word in ["job", "work", "career", "position", "hiring"]):
        return QUICK_RESPONSES["job"]
    elif any(word in user_lower for word in ["course", "learn", "study", "education", "training"]):
        return QUICK_RESPONSES["course"]
    elif any(word in user_lower for word in ["mentor", "guidance", "advice", "coach"]):
        return QUICK_RESPONSES["mentorship"]
    elif any(word in user_lower for word in ["assessment", "test", "skill", "evaluate"]):
        return QUICK_RESPONSES["assessment"]
    else:
        # Default helpful response
        return f"""🤔 **I'd love to help you with that!**

Here are the main areas I can assist with:

💼 **Jobs**: https://www.impacteers.com/jobs - Find your next opportunity
📚 **Courses**: https://www.impacteers.com/courses - Learn new skills  
🎯 **Assessments**: https://www.impacteers.com/assessments - Discover your strengths
👥 **Mentorship**: https://www.impacteers.com/mentorship - Get expert guidance
🌟 **Community**: https://www.impacteers.com/community - Connect with peers

💡 **For specific questions about "{user_input}", I recommend:**
1. Browse our knowledge base at https://www.impacteers.com
2. Join our community for peer discussions
3. Book a mentorship session for personalized advice

What would you like to explore first?"""

# Chat interface
st.subheader("💬 Chat with Impacteers AI")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🎯 Impacteers AI:</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Type your message...", 
        placeholder="Ask about jobs, courses, mentorship, or anything career-related!",
        key="chat_input"
    )

with col2:
    send_button = st.button("Send 🚀")

# Process user input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.spinner("🤖 Thinking..."):
        response = get_ai_response(user_input)
    
    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to show new messages
    st.rerun()

# Sidebar with quick actions
with st.sidebar:
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("💼 Find Jobs"):
        st.session_state.messages.append({"role": "user", "content": "I'm looking for jobs"})
        st.session_state.messages.append({"role": "assistant", "content": QUICK_RESPONSES["job"]})
        st.rerun()
    
    if st.button("📚 Explore Courses"):
        st.session_state.messages.append({"role": "user", "content": "Show me courses"})
        st.session_state.messages.append({"role": "assistant", "content": QUICK_RESPONSES["course"]})
        st.rerun()
    
    if st.button("👥 Get Mentorship"):
        st.session_state.messages.append({"role": "user", "content": "I need mentorship"})
        st.session_state.messages.append({"role": "assistant", "content": QUICK_RESPONSES["mentorship"]})
        st.rerun()
    
    if st.button("🎯 Take Assessment"):
        st.session_state.messages.append({"role": "user", "content": "I want to take an assessment"})
        st.session_state.messages.append({"role": "assistant", "content": QUICK_RESPONSES["assessment"]})
        st.rerun()
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("User ID", st.session_state.user_id)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎯 <strong>Impacteers AI Career Assistant</strong> | Powered by Streamlit Cloud</p>
    <p>🚀 <a href="https://www.impacteers.com" target="_blank">Visit Impacteers.com</a> | 
       📧 <a href="mailto:support@impacteers.com">Get Support</a></p>
</div>
""", unsafe_allow_html=True)