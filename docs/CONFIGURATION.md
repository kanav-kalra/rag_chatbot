# Configuration Guide

This guide covers all configuration options for the RAG Chatbot application, including environment variables, YAML configuration files, and settings.

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Project Information

```bash
PROJECT_NAME="RAG Chatbot"
PROJECT_VERSION="1.0.0"
```

### API Configuration

```bash
HOST=0.0.0.0
PORT=8000
STREAMLIT_PORT=8501
DEBUG=False
```

### API Keys (Required)

```bash
# Required for OpenAI embeddings and LLM
OPENAI_API_KEY=your-openai-api-key-here

# Optional: For Anthropic Claude
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional: For Google Gemini
GOOGLE_API_KEY=your-google-api-key
```

### Redis Configuration

```bash
# For session checkpointing
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Session Management

```bash
MAX_CONCURRENT_SESSIONS=1000
SESSION_TIMEOUT_HOURS=24
```

### Agent Pool Configuration

```bash
AGENT_POOL_SIZE=1  # Number of shared agents (default: 1)
```

### LLM Configuration (Defaults)

```bash
CHAT_MODEL=gpt-4
CHAT_MODEL_TEMPERATURE=0.7
CHAT_MODEL_MAX_TOKENS=2000
```

### Logging

```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Chatbot Configuration Files

Each chatbot type has its own YAML configuration file in `config/chatbot/`. For example, `hr_chatbot_config.yaml`:

### Model Configuration

```yaml
model:
  name: "gemini-2.5-flash"  # LLM model name
  temperature: 0.2           # Temperature (0.0-2.0)
  max_tokens: 2000           # Maximum tokens in response
  base_url: null             # Optional, for Ollama or custom endpoints
```

**Supported Models**:
- OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Google: `gemini-2.5-flash`, `gemini-pro`
- Ollama: Any Ollama model (requires `base_url`)

### Vector Store Configuration

```yaml
vector_store:
  type: "hr"  # Unique identifier (must match chatbot type)
  persist_dir: "./data/vectorstores/chroma_db/hr_chatbot"
  collection_name: "hr_chatbot"
  embedding_provider: "auto"  # "auto", "openai", or "google"
  embedding_model: ""  # Empty = use provider default
```

**Embedding Providers**:
- `auto`: Automatically detects based on model name
- `openai`: Uses OpenAI embeddings (requires `OPENAI_API_KEY`)
- `google`: Uses Google embeddings (requires `GOOGLE_API_KEY`)

**Embedding Models**:
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- Google: `models/embedding-001`

### System Prompt Configuration

```yaml
system_prompt:
  prompts_file: "hr_chatbot_prompts.yaml"  # Prompts file name
  template: null  # If null, uses system_prompt from prompts_file
  agent_instructions_template: null  # If null, uses agent_instructions from prompts_file
```

### Tools Configuration

```yaml
tools:
  enable_retrieval: true  # Enable document retrieval tool
  additional: []  # Additional tools beyond retrieval
```

### Memory Configuration

```yaml
memory:
  strategy: "trim"  # Options: "none", "trim", "summarize", "trim_and_summarize"
  trim_keep_messages: 1  # Keep last N messages when trimming
  summarize_threshold: 2  # Summarize when messages exceed this count
  summarize_model: "gpt-3.5-turbo-16k"  # Model for summarization
```

**Memory Strategies**:
- `none`: Keep all messages (may hit context limits)
- `trim`: Keep only last N messages
- `summarize`: Summarize old messages when threshold reached
- `trim_and_summarize`: Combine trim and summarize

### Agent Pool Configuration

```yaml
agent_pool:
  size: 1  # Number of shared agents (default: 1)
```

**Pool Size Guidelines**:
- `1`: Best for most cases, single shared agent
- `2-5`: For moderate concurrency, reduces contention
- `10+`: For very high concurrency (may be overkill)

### Verbose Logging

```yaml
verbose: false  # Enable verbose logging for debugging
```

## Prompts Configuration

Prompts are stored in YAML files in `config/chatbot/prompts/`. For example, `hr_chatbot_prompts.yaml`:

```yaml
system_prompt: |
  You are a professional HR Assistant. Your primary goal is to provide
  accurate information based ONLY on the provided document context.

agent_instructions: |
  You are an HR Assistant Agent. Follow this execution flow:
  1. Retrieve: Always call the retrieve_documents tool first
  2. Filter: Select only the most relevant snippets
  3. Formulate: Write a concise answer using the snippets
  4. Verify: Check if your answer contains information NOT in the context

rag_prompt_template: |
  Use the context below to answer.
  Context: {context}
  Question: {question}
  Answer:

topic_prompts:
  leave_policy: |
    You are answering a question about leave policies...
  benefits: |
    You are answering a question about employee benefits...
```

## Environment Variable Overrides

Many configuration values can be overridden via environment variables. The naming convention is:

```
{CHATBOT_TYPE}_{CONFIG_KEY}
```

For example, for HR chatbot:
- `HR_CHAT_MODEL` overrides `model.name`
- `HR_CHAT_MODEL_TEMPERATURE` overrides `model.temperature`
- `HR_AGENT_POOL_SIZE` overrides `agent_pool.size`

## Configuration Priority

1. **Environment Variables** (highest priority)
2. **YAML Configuration File**
3. **Default Values** (lowest priority)

## Example: Complete Configuration

### `.env` file

```bash
# Project
PROJECT_NAME="RAG Chatbot"
PROJECT_VERSION="1.0.0"

# API
HOST=0.0.0.0
PORT=8000
STREAMLIT_PORT=8501
DEBUG=False

# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Session
MAX_CONCURRENT_SESSIONS=1000
SESSION_TIMEOUT_HOURS=24

# Agent Pool
AGENT_POOL_SIZE=1

# Logging
LOG_LEVEL=INFO
```

### `hr_chatbot_config.yaml`

```yaml
model:
  name: "gemini-2.5-flash"
  temperature: 0.2
  max_tokens: 2000

vector_store:
  type: "hr"
  persist_dir: "./data/vectorstores/chroma_db/hr_chatbot"
  collection_name: "hr_chatbot"
  embedding_provider: "auto"

system_prompt:
  prompts_file: "hr_chatbot_prompts.yaml"

tools:
  enable_retrieval: true

memory:
  strategy: "trim"
  trim_keep_messages: 1
  summarize_threshold: 2

agent_pool:
  size: 1

verbose: false
```

## Configuration Validation

The application validates configuration on startup:
- Required environment variables are checked
- YAML files are validated for syntax
- Configuration values are checked for valid ranges
- Dependencies are verified (e.g., API keys for selected providers)

## Troubleshooting

### Configuration Not Loading

- Check file paths are correct
- Verify YAML syntax is valid
- Check environment variables are set
- Review logs for configuration errors

### API Key Errors

- Verify API keys are set in `.env`
- Check keys are valid and not expired
- Ensure correct provider keys for selected models

### Vector Store Not Found

- Verify `persist_dir` path exists
- Check collection name matches
- Ensure vector store was created

### Memory Issues

- Check memory strategy configuration
- Verify summarize model has sufficient context window
- Monitor session count and memory usage

## Related Documentation

- [Creating a New Chatbot](CREATING_NEW_CHATBOT.md) - Configuration for new chatbots
- [Session Management](SESSION_MANAGEMENT.md) - Session configuration
- [Vector Store Management](VECTOR_STORE.md) - Vector store configuration

