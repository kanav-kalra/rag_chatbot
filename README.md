# RAG chatbot

A Python application combining FastAPI backend with Streamlit frontend for RAG (Retrieval-Augmented Generation) chatbot functionality.

## Project Structure

```
app/
├── main.py                 # Main entry point (FastAPI + Streamlit launcher)
├── streamlit_app.py        # Streamlit main page
├── pages/                  # Streamlit multi-page structure
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔧_API_Explorer.py
│   └── 3_⚙️_Settings.py
├── api/                    # FastAPI routes
└── core/                   # Configuration
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Method 1: Command Line

Run both applications (default):
```bash
python -m app.main
# or
python -m app.main --app both
```

Run only FastAPI:
```bash
python -m app.main --app fastapi
```

Run only Streamlit:
```bash
python -m app.main --app streamlit
```

### Method 2: Using Cursor/VS Code

1. **Using Launch Configuration** (Recommended):
   - Press `F5` or go to Run and Debug
   - Select one of the pre-configured options:
     - "Python: Run Both (FastAPI + Streamlit)"
     - "Python: Run FastAPI Only"
     - "Python: Run Streamlit Only"

2. **Using Integrated Terminal**:
   - Open the integrated terminal (`` Ctrl+` ``)
   - Run the commands from Method 1

3. **Using Run Button**:
   - Right-click on `app/main.py`
   - Select "Run Python File in Terminal"
   - Add arguments in the terminal if needed

### Method 3: Direct Module Execution

```bash
# Run both
python -m app.main --app both

# Run FastAPI only
python -m app.main --app fastapi

# Run Streamlit only
python -m app.main --app streamlit
```

## Accessing the Applications

- **FastAPI**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501

## Environment Variables

You can configure the application using environment variables:

```bash
export PROJECT_NAME="My Application"
export PORT=8000
export STREAMLIT_PORT=8501
export DEBUG=True
export HOST=0.0.0.0
```

## Adding New Streamlit Pages

To add a new page, create a file in `app/pages/` with the format:
```
N_🎯_Page_Name.py
```

Where:
- `N` is a number for ordering
- `🎯` is an emoji icon (optional)
- `Page_Name` is the page name (use underscores)

Example: `4_📝_Notes.py`
