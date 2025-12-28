"""
Chatbot Page - Interactive HR Chatbot Interface
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import requests
from app.core.config import settings

st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 HR Chatbot")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# API endpoint
API_URL = "http://127.0.0.1:8000/api/v1/chat"

# Display chat history in a scrollable container
st.subheader("Chat History")

# Create a scrollable container for chat messages
chat_container = st.container(height=500)

# Display all messages in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Reset button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat input - handles Enter key automatically
user_query = st.chat_input("Type your question here and press Enter...")

# Handle user input (from chat_input - Enter key or button click)
if user_query:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    # Show loading indicator and get response
    with st.spinner("Thinking..."):
        try:
            # Call API with just the latest query
            payload = {
                "message": user_query
            }
            
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            # Get response from API
            api_response = response.json()
            response_text = api_response.get("response", "No response received")
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to API: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {error_msg}"
            })
            st.error(error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {error_msg}"
            })
            st.error(error_msg)
    
    # Rerun to update the display (will show all messages including new ones)
    st.rerun()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This is an HR chatbot that can answer questions about HR policies and documents.")
    st.divider()
    
    st.header("💡 Tips")
    st.write("• Ask questions about HR policies")
    st.write("• Use the Reset button to clear chat history")
    st.write("• Chat history is maintained during your session")
    st.write("• Press Enter or click Submit to send your message")
    
    st.divider()
    
    st.header("🔗 API Connection")
    st.write(f"**Endpoint:** {API_URL}")
    
    # Test API connection
    try:
        response = requests.get(f"http://127.0.0.1:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API may not be ready")
    except:
        st.error("❌ API Not Available")
