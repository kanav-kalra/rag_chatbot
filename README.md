# RAG Chatbot

A Python application combining FastAPI backend with Streamlit frontend for RAG (Retrieval-Augmented Generation) chatbot functionality.

## Project Structure

The project follows clean architecture principles with clear separation of concerns:

```
rag_chatbot/
├── src/                          # Main application source code
│   ├── main.py                   # Application entry point (FastAPI + Streamlit launcher)
│   ├── domain/                   # Domain layer (business logic, entities)
│   │   ├── chatbot/             # Chatbot domain
│   │   │   ├── core/            # Core chatbot framework
│   │   │   │   ├── agent.py     # ChatbotAgent base class
│   │   │   │   ├── config.py    # ChatbotConfigManager
│   │   │   │   ├── prompts.py   # ChatbotPromptBuilder
│   │   │   │   └── tools.py     # ChatbotToolFactory
│   │   │   └── hr_chatbot.py    # HRChatbot implementation
│   │   ├── memory/              # Memory domain
│   │   ├── retrieval/           # Retrieval domain
│   │   └── session/             # Session domain
│   ├── application/             # Application layer (use cases, orchestration)
│   │   ├── chatbot/             # Chatbot use cases
│   │   └── ingestion/           # Ingestion use cases
│   ├── infrastructure/          # Infrastructure layer (external concerns)
│   │   ├── llm/                 # LLM providers
│   │   ├── vectorstore/         # Vector store implementations
│   │   └── storage/             # Storage implementations
│   ├── api/                     # API layer (presentation)
│   │   ├── middleware/          # API middleware
│   │   └── v1/                  # API v1 routes
│   ├── ui/                      # UI layer (Streamlit)
│   │   └── pages/               # Streamlit pages
│   └── shared/                  # Shared utilities
│       ├── config/               # Configuration
│       ├── memory/               # Memory config
│       └── dependencies/         # Dependency injection
├── scripts/                      # Scripts and tools
│   ├── ingestion/               # Ingestion scripts
│   └── jobs/                    # Background jobs
├── config/                       # Configuration files
│   └── chatbot/                 # Chatbot configs
│       └── prompts/              # Prompt templates
├── data/                         # Data directories
│   ├── vectorstores/             # Vector store data
│   └── logs/                     # Application logs
├── evaluations/                  # Evaluation code
└── docs/                         # Documentation
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
python -m src.main
# or
python -m src.main --app both
```

Run only FastAPI:
```bash
python -m src.main --app fastapi
```

Run only Streamlit:
```bash
python -m src.main --app streamlit
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
   - Right-click on `src/main.py`
   - Select "Run Python File in Terminal"
   - Add arguments in the terminal if needed

### Method 3: Direct Module Execution

```bash
# Run both
python -m src.main --app both

# Run FastAPI only
python -m src.main --app fastapi

# Run Streamlit only
python -m src.main --app streamlit
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

## Architecture

The project follows **Clean Architecture** principles:

- **Domain Layer** (`src/domain/`): Core business logic, entities, and domain models. No dependencies on external frameworks.
- **Application Layer** (`src/application/`): Use cases and orchestration logic. Coordinates domain objects.
- **Infrastructure Layer** (`src/infrastructure/`): External services (LLM providers, vector stores, storage).
- **API Layer** (`src/api/`): FastAPI routes and middleware.
- **UI Layer** (`src/ui/`): Streamlit application.
- **Shared** (`src/shared/`): Common utilities and configuration.

## Adding New Streamlit Pages

To add a new page, create a file in `src/ui/pages/` with the format:
```
dashboard.py
api_explorer.py
settings.py
chatbot.py
```

Pages are automatically discovered by Streamlit based on filename.

## Configuration

Chatbot configurations are stored in YAML files in `config/chatbot/`:
- `hr_chatbot_config.yaml`: HR chatbot configuration
- `prompts/default.yaml`: Default prompts
- `prompts/hr_chatbot.yaml`: HR chatbot prompts

## Data Directories

- `data/vectorstores/`: Vector store data (ChromaDB)
- `data/logs/`: Application logs

## Scripts

Ingestion and utility scripts are in `scripts/`:
- `scripts/ingestion/`: Scripts for creating and managing vector stores
- `scripts/jobs/`: Background job scripts

## Migration from Old Structure

If you're migrating from the old `app/` structure, see `MIGRATION_GUIDE.md` for details.

## Documentation

- `MIGRATION_GUIDE.md`: Guide for migrating from old structure
- `docs/guides/`: Architecture and usage guides
- `docs/design/`: Design documents
