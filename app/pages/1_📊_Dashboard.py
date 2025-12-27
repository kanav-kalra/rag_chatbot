"""
Dashboard Page
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.core.config import settings

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("API Status", "Running", "✓")
    
with col2:
    st.metric("Version", settings.PROJECT_VERSION)
    
with col3:
    st.metric("Port", settings.PORT)

st.divider()

st.subheader("Quick Links")
col1, col2 = st.columns(2)

with col1:
    st.link_button(
        "API Documentation",
        f"http://{settings.HOST}:{settings.PORT}/docs",
        use_container_width=True
    )

with col2:
    st.link_button(
        "API Health Check",
        f"http://{settings.HOST}:{settings.PORT}/health",
        use_container_width=True
    )

