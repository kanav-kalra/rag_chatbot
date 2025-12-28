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

# Required for OpenAI embeddings and vector store functionality
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Setting OPENAI_API_KEY

The `OPENAI_API_KEY` is required for the document loader's memory builder functionality. You can set it in several ways:

1. **Using export command (temporary - for current terminal session):**
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

2. **Using .env file (recommended):**
   Create a `.env` file in the project root:
   ```bash
   echo 'OPENAI_API_KEY=sk-your-api-key-here' > .env
   ```
   Then load it in your shell:
   ```bash
   source .env  # Note: this won't auto-load, you need to export it
   export $(cat .env | xargs)
   ```
   
   Or install `python-dotenv` and load it programmatically:
   ```bash
   pip install python-dotenv
   ```

3. **In your shell profile (persistent):**
   Add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`:
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

**Note:** The `.env` file is already in `.gitignore` to keep your API key secure. Never commit your API key to version control.

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
