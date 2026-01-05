# Creating a New Chatbot

This guide walks you through creating a new chatbot type, similar to the HR chatbot. The process is straightforward thanks to the base `ChatbotAgent` class that handles all the heavy lifting.

## Overview

Creating a new chatbot requires:
1. Configuration file (YAML)
2. Prompts file (YAML)
3. Chatbot class (Python)
4. Vector store creation
5. Optional: API route

The base `ChatbotAgent` class handles:
- Configuration loading
- Tool creation (retrieval, etc.)
- Prompt building
- Memory management
- Agent pool management

## Step 1: Create Configuration File

Create a YAML configuration file in `config/chatbot/`. For example, `support_chatbot_config.yaml`:

```yaml
# Model Configuration
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  base_url: null  # Optional, mainly for Ollama

# Vector Store Configuration
vector_store:
  type: "support"  # Unique identifier for this chatbot
  persist_dir: "./data/vectorstores/chroma_db/support_chatbot"
  collection_name: "support_chatbot"
  embedding_provider: "auto"  # "auto", "openai", or "google"
  embedding_model: ""  # Empty = use provider default

# System Prompt Configuration
system_prompt:
  prompts_file: "support_chatbot_prompts.yaml"  # Create this file next
  template: null  # If null, uses system_prompt from prompts_file
  agent_instructions_template: null  # If null, uses agent_instructions from prompts_file

# Tools Configuration
tools:
  enable_retrieval: true  # Enable document retrieval tool
  additional: []  # Additional tools beyond retrieval

# Memory Configuration
memory:
  strategy: "trim"  # Options: "none", "trim", "summarize", "trim_and_summarize"
  trim_keep_messages: 1  # Keep last N messages when trimming
  summarize_threshold: 2  # Summarize when messages exceed this count
  summarize_model: "gpt-3.5-turbo-16k"  # Model for summarization

# Agent Pool Configuration
agent_pool:
  size: 1  # Number of shared agents (default: 1)

# Verbose Logging
verbose: false
```

### Configuration Options

- **model.name**: LLM model to use (e.g., "gpt-4", "gemini-2.5-flash", "claude-3-opus")
- **vector_store.type**: Unique identifier (must match chatbot type)
- **vector_store.persist_dir**: Where to store ChromaDB data
- **tools.enable_retrieval**: Enable RAG functionality
- **memory.strategy**: How to manage conversation history

## Step 2: Create Prompts File

Create a prompts file in `config/chatbot/prompts/`. For example, `support_chatbot_prompts.yaml`:

```yaml
system_prompt: |
  You are a helpful Support Assistant. Your primary goal is to assist users
  with technical questions and troubleshooting.

  OPERATING RULES:
  1. Be friendly and professional
  2. Provide step-by-step solutions when possible
  3. If you don't know the answer, direct users to appropriate resources
  4. Include relevant code examples when applicable

agent_instructions: |
  You are a Support Assistant Agent. Follow this execution flow:
  
  1. Retrieve: Always call the retrieve_documents tool first
  2. Analyze: Review the retrieved context carefully
  3. Respond: Provide a clear, helpful answer based on the context
  4. Verify: Ensure your answer is accurate and complete
  5. Format: Use code blocks for code examples
```

### Prompt Structure

- **system_prompt**: Overall behavior and tone
- **agent_instructions**: Step-by-step execution flow
- **rag_prompt_template**: Optional RAG-specific template
- **topic_prompts**: Optional topic-specific prompts

## Step 3: Create Chatbot Class

Create a new file `src/domain/chatbot/support_chatbot.py`:

```python
"""
Support Chatbot - Implementation using refactored ChatbotAgent architecture.
All configuration comes from support_chatbot_config.yaml via ChatbotConfigManager.
"""
from src.domain.chatbot.core.chatbot_agent import ChatbotAgent


class SupportChatbot(ChatbotAgent):
    """
    Support-specific chatbot implementation.
    
    This is a minimal subclass that only defines:
    1. Chatbot type identifier ("support")
    2. Config filename ("support_chatbot_config.yaml")
    
    All other functionality is handled by the base ChatbotAgent class.
    """
    
    def _get_chatbot_type(self) -> str:
        """Return the chatbot type identifier."""
        return "support"
    
    @classmethod
    def _get_chatbot_type_class(cls) -> str:
        """Return the chatbot type identifier without instantiation."""
        return "support"
    
    @classmethod
    def _get_config_filename(cls) -> str:
        """Return the YAML config filename."""
        return "support_chatbot_config.yaml"
    
    @classmethod
    def _get_default_instance(cls) -> "SupportChatbot":
        """Create a default Support chatbot instance for the agent pool."""
        return SupportChatbot()


def get_support_chatbot() -> SupportChatbot:
    """
    Get a Support chatbot instance from the agent pool.
    
    Returns:
        SupportChatbot instance from agent pool
    """
    return SupportChatbot.get_from_pool()


__all__ = [
    "SupportChatbot",
    "get_support_chatbot",
]
```

### Required Methods

- **`_get_chatbot_type()`**: Return chatbot type (must match `vector_store.type` in config)
- **`_get_chatbot_type_class()`**: Class method version (for efficiency)
- **`_get_config_filename()`**: Return config filename
- **`_get_default_instance()`**: Create instance for agent pool

### Optional Overrides

You can override these methods for custom behavior:
- **`_get_prompts_filename()`**: Custom prompts file name
- **`_create_tools()`**: Custom tool creation
- **`_create_system_prompt()`**: Custom prompt building

## Step 4: Create Vector Store

### Option A: Using Script

Create `scripts/jobs/create_support_vectorstore.sh`:

```bash
#!/bin/bash
# Shell script wrapper for creating Support chatbot vector store
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
python scripts/ingestion/create_vectorstore.py --chatbot-type support "$@"
```

Make it executable:
```bash
chmod +x scripts/jobs/create_support_vectorstore.sh
```

Run it:
```bash
./scripts/jobs/create_support_vectorstore.sh --folder-path /path/to/your/documents
```

### Option B: Direct Command

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type support \
  --folder-path /path/to/your/documents \
  --chunk-size 1000 \
  --chunk-overlap 200
```

### Vector Store Options

- **`--folder-path`**: Path to folder containing PDFs
- **`--chunk-size`**: Maximum chunk size (default: 1000)
- **`--chunk-overlap`**: Overlap between chunks (default: 200)
- **`--recursive`**: Search subdirectories (default: true)
- **`--clear-existing`**: Delete existing collection first
- **`--skip-if-exists`**: Skip if collection already exists

## Step 5: Create API Route (Optional)

If you want a dedicated API endpoint, add a route in `src/api/v1/routes/chat.py`:

```python
from src.domain.chatbot.support_chatbot import get_support_chatbot

@router.post("/support", response_model=ChatResponse, tags=["chat"])
async def chat_with_support_chatbot(
    request: ChatRequest,
    session: ChatbotSession = Depends(get_session_from_headers)
):
    """
    Chat with the Support chatbot.
    
    Send a message to the Support chatbot and receive a response based on
    support documentation and knowledge base.
    """
    try:
        chatbot = get_support_chatbot()
        
        # Get or create thread ID from session
        thread_id = session.thread_id
        
        # Chat with the chatbot
        response = chatbot.chat(request.message, thread_id=thread_id)
        
        # Update session
        session_manager = get_session_manager_dependency()
        session_manager.update_session_activity(session.session_id)
        
        return ChatResponse(
            response=response,
            session_id=session.session_id,
            model_used=chatbot.config_manager.get("model.name", "unknown"),
            message_count=session.message_count
        )
    except Exception as e:
        logger.error(f"Error in support chatbot: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get response from support chatbot: {str(e)}"
        )
```

## Step 6: Test Your Chatbot

### Test Directly

```python
from src.domain.chatbot.support_chatbot import get_support_chatbot

# Get chatbot from pool
chatbot = get_support_chatbot()

# Chat with it
response = chatbot.chat("How do I reset my password?", thread_id="test-123")
print(response)
```

### Test via API

```bash
curl -X POST "http://localhost:8000/api/v1/chat/support" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

### Test in Streamlit

Add a page in `src/ui/pages/support_chatbot.py` (similar to `chatbot.py`).

## Complete Example: Customer Service Chatbot

Here's a complete example for a customer service chatbot:

### 1. Config: `customer_service_chatbot_config.yaml`

```yaml
model:
  name: "gpt-4"
  temperature: 0.5
  max_tokens: 2000

vector_store:
  type: "customer_service"
  persist_dir: "./data/vectorstores/chroma_db/customer_service_chatbot"
  collection_name: "customer_service_chatbot"
  embedding_provider: "auto"

system_prompt:
  prompts_file: "customer_service_chatbot_prompts.yaml"

tools:
  enable_retrieval: true

memory:
  strategy: "trim"
  trim_keep_messages: 3

agent_pool:
  size: 1
```

### 2. Prompts: `customer_service_chatbot_prompts.yaml`

```yaml
system_prompt: |
  You are a Customer Service Representative. Your goal is to help customers
  with their inquiries, resolve issues, and provide excellent service.

agent_instructions: |
  1. Retrieve relevant information from knowledge base
  2. Provide clear, helpful answers
  3. Escalate complex issues when needed
```

### 3. Class: `src/domain/chatbot/customer_service_chatbot.py`

```python
from src.domain.chatbot.core.chatbot_agent import ChatbotAgent

class CustomerServiceChatbot(ChatbotAgent):
    def _get_chatbot_type(self) -> str:
        return "customer_service"
    
    @classmethod
    def _get_chatbot_type_class(cls) -> str:
        return "customer_service"
    
    @classmethod
    def _get_config_filename(cls) -> str:
        return "customer_service_chatbot_config.yaml"
    
    @classmethod
    def _get_default_instance(cls):
        return CustomerServiceChatbot()

def get_customer_service_chatbot():
    return CustomerServiceChatbot.get_from_pool()
```

### 4. Vector Store

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type customer_service \
  --folder-path /path/to/customer_service_docs
```

## Troubleshooting

### Configuration Not Found

- Ensure config file is in `config/chatbot/`
- Check filename matches `_get_config_filename()` return value
- Verify YAML syntax is correct

### Vector Store Not Found

- Ensure vector store was created
- Check `persist_dir` in config matches actual location
- Verify collection name matches

### Import Errors

- Ensure chatbot class is in `src/domain/chatbot/`
- Check `__init__.py` exports the class
- Verify Python path includes project root

### Agent Not Working

- Check logs in `data/logs/app.log`
- Verify API keys are set
- Test with verbose logging enabled

## Best Practices

1. **Naming**: Use descriptive, consistent names (e.g., `support_chatbot`, not `bot1`)
2. **Config**: Keep configs in YAML, avoid hardcoding
3. **Prompts**: Write clear, specific prompts for better responses
4. **Testing**: Test with various queries before deploying
5. **Documentation**: Document any custom behavior

## Related Documentation

- [Architecture Guide](ARCHITECTURE.md) - System architecture
- [Configuration Guide](CONFIGURATION.md) - Configuration details
- [Vector Store Management](VECTOR_STORE.md) - Vector store operations
- [API Usage Guide](API_USAGE.md) - API integration

