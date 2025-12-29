"""
API Explorer Page
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
    page_title="API Explorer",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 API Explorer")

st.write("Test and explore your FastAPI endpoints")

# API Base URL
api_base_url = f"http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}"

st.subheader("Available Endpoints")

# Health Check
with st.expander("Health Check", expanded=False):
    st.code(f"GET {api_base_url.replace(settings.API_PREFIX, '')}/health")
    if st.button("Test Health Check", key="health"):
        try:
            response = requests.get(f"http://{settings.HOST}:{settings.PORT}/health")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")

# Root Endpoint
with st.expander("Root Endpoint", expanded=False):
    st.code(f"GET {api_base_url.replace(settings.API_PREFIX, '')}/")
    if st.button("Test Root", key="root"):
        try:
            response = requests.get(f"http://{settings.HOST}:{settings.PORT}/")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")

st.info(f"💡 Visit [API Documentation](http://{settings.HOST}:{settings.PORT}/docs) for interactive API testing")

