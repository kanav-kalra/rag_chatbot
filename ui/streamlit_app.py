"""
Streamlit Application - Main Page
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.core.config import settings
from app.core.langsmith_config import initialize_langsmith

# Initialize LangSmith tracing (if enabled)
# This ensures LangSmith works when Streamlit is run independently
initialize_langsmith()

st.set_page_config(
    page_title=settings.PROJECT_NAME,
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"🚀 Welcome to {settings.PROJECT_NAME}")
st.write(f"**Version:** {settings.PROJECT_VERSION}")

st.divider()

st.header("About")
st.write(settings.PROJECT_DESCRIPTION)

st.divider()

st.header("Quick Start")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 FastAPI")
    st.write(f"**API Server:** http://{settings.HOST}:{settings.PORT}")
    st.write(f"**API Documentation:** http://{settings.HOST}:{settings.PORT}/docs")
    st.write(f"**ReDoc:** http://{settings.HOST}:{settings.PORT}/redoc")
    
    if st.button("Open API Docs", use_container_width=True):
        st.markdown(f"[Open in new tab](http://{settings.HOST}:{settings.PORT}/docs)")

with col2:
    st.subheader("📊 Streamlit")
    st.write("Navigate using the sidebar to explore different pages:")
    st.write("• **Dashboard** - Overview and metrics")
    st.write("• **API Explorer** - Test API endpoints")
    st.write("• **Settings** - Application configuration")
    st.write("• **Chatbot** - Interactive HR chatbot")

st.divider()

st.info("💡 Use the sidebar to navigate between different pages of the application.")

