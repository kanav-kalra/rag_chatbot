# RAG Chatbot

A production-ready Python application combining FastAPI backend with Streamlit frontend for RAG (Retrieval-Augmented Generation) chatbot functionality. Built with clean architecture principles, supporting multiple chatbot types with shared agent pools for efficient resource management.

## Quick Start

1. **Install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env-sample .env
   # Edit .env and add your API keys
   ```

3. **Run the application**:
   ```bash
   python -m src.main
   ```

4. **Access the applications**:
   - FastAPI: http://localhost:8000
   - FastAPI Docs: http://localhost:8000/docs
   - Streamlit: http://localhost:8501

## Features

- ✅ **Multi-Chatbot Support**: Easy to create new chatbot types (HR, Support, etc.)
- ✅ **RAG (Retrieval-Augmented Generation)**: Document-based Q&A with vector search
- ✅ **Session Management**: Automatic session handling with Redis checkpointing
- ✅ **Shared Agent Pool**: Efficient resource usage across multiple concurrent users
- ✅ **Clean Architecture**: Separation of concerns with domain, application, infrastructure layers
- ✅ **FastAPI Backend**: RESTful API with automatic documentation
- ✅ **Streamlit Frontend**: Interactive UI for testing and demonstration
- ✅ **Memory Management**: Configurable memory strategies (trim, summarize, etc.)
- ✅ **Multiple LLM Support**: OpenAI, Anthropic, Google (Gemini), Ollama
- ✅ **Vector Store**: ChromaDB with persistent storage

## Documentation

### Core Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: System architecture, layer responsibilities, and design principles
- **[HR Chatbot Flow](docs/HR_CHATBOT_FLOW.md)**: Detailed flow diagram and explanation of the HR chatbot request processing
- **[Creating a New Chatbot](docs/CREATING_NEW_CHATBOT.md)**: Step-by-step guide to create a new chatbot type
- **[API Usage Guide](docs/API_USAGE.md)**: API endpoints, examples, and client usage
- **[Configuration Guide](docs/CONFIGURATION.md)**: Environment variables, YAML configs, and settings
- **[Vector Store Management](docs/VECTOR_STORE.md)**: Creating, managing, and updating vector stores
- **[Session Management](docs/SESSION_MANAGEMENT.md)**: Session lifecycle, headers, and shared agent pool
- **[Evaluation Guide](docs/EVALUATION.md)**: How to evaluate chatbots using LLM-as-Judge with LangSmith

### Additional Guides

- **[Shared Agent Pool](SHARED_AGENT_POOL.md)**: Details about the shared agent pool architecture
- **[Memory Management](MEMORY_MANAGEMENT_GUIDE.md)**: Memory management strategies and configuration
- **[Retrieval Service Design](RETRIEVAL_SERVICE_DESIGN.md)**: Retrieval service architecture
- **[LangGraph Studio Guide](LANGGRAPH_STUDIO_GUIDE.md)**: Using LangGraph Studio for debugging

## Project Structure

```
rag_chatbot/
├── src/                          # Main application source code
│   ├── main.py                   # Application entry point
│   ├── domain/                   # Domain layer (business logic)
│   ├── application/              # Application layer (use cases)
│   ├── infrastructure/           # Infrastructure layer (external services)
│   ├── api/                      # API layer (FastAPI)
│   ├── ui/                       # UI layer (Streamlit)
│   └── shared/                   # Shared utilities
├── scripts/                      # Scripts and tools
├── config/                       # Configuration files
├── data/                         # Data directories
├── evaluations/                  # Evaluation code
└── docs/                         # Documentation
```

See [Architecture Guide](docs/ARCHITECTURE.md) for detailed structure.

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd rag_chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env-sample .env
   # Edit .env and add your API keys (OPENAI_API_KEY, etc.)
   ```

## Running the Application

### Command Line

```bash
# Run both applications (default)
python -m src.main

# Run only FastAPI
python -m src.main --app fastapi

# Run only Streamlit
python -m src.main --app streamlit
```

### Using Cursor/VS Code

1. Press `F5` or go to Run and Debug
2. Select one of the pre-configured options:
   - "Python: Run Both (FastAPI + Streamlit)"
   - "Python: Run FastAPI Only"
   - "Python: Run Streamlit Only"

### Accessing the Applications

- **FastAPI**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501

## Quick Examples

### Using the API

```python
import requests

# Chat with HR chatbot
response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={"message": "What is the vacation policy?"}
)
print(response.json()["response"])
```

### Using the Chatbot Directly

```python
from src.domain.chatbot.hr_chatbot import get_hr_chatbot

chatbot = get_hr_chatbot()
response = chatbot.chat("Hello", thread_id="test-123")
print(response)
```

## Next Steps

1. **Read the [Architecture Guide](docs/ARCHITECTURE.md)** to understand the system design
2. **Check the [HR Chatbot Flow](docs/HR_CHATBOT_FLOW.md)** to see how requests are processed
3. **Follow [Creating a New Chatbot](docs/CREATING_NEW_CHATBOT.md)** to build your own chatbot
4. **Review [API Usage Guide](docs/API_USAGE.md)** for API integration examples
5. **Learn [Evaluation Guide](docs/EVALUATION.md)** to evaluate your chatbot performance

## Troubleshooting

### Common Issues

1. **Vector Store Not Found**
   - Ensure you've created the vector store (see [Vector Store Management](docs/VECTOR_STORE.md))
   - Check that `persist_dir` in config matches the actual location

2. **API Key Errors**
   - Verify your API keys are set in `.env` file
   - See [Configuration Guide](docs/CONFIGURATION.md) for details

3. **Session Not Persisting**
   - Ensure Redis is running if using Redis checkpointing
   - Check [Session Management](docs/SESSION_MANAGEMENT.md) guide

4. **Import Errors**
   - Ensure you're running from the project root
   - Verify virtual environment is activated

### Getting Help

- Check the logs in `data/logs/app.log`
- Review API documentation at http://localhost:8000/docs
- Check configuration files in `config/chatbot/`
- See detailed guides in the `docs/` directory

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
