# Memory Management with LangChain Checkpointer

## Overview

The chatbot system has been updated to use LangChain's checkpointer with Redis for short-term memory storage, replacing the previous session manager-based chat history. The system now supports configurable memory management strategies (trim, summarize, or combination) for each chatbot.

## Key Changes

### 1. Redis Checkpointer Integration
- Replaced session manager's in-memory chat history with LangChain's checkpointer
- Uses Redis for persistent short-term memory storage
- Falls back to in-memory checkpointer if Redis is unavailable

### 2. Configurable Memory Management
Each chatbot (e.g., HR chatbot) can be configured with different memory strategies:
- **none**: No memory management (keep all messages)
- **trim**: Trim old messages, keep only recent ones
- **summarize**: Summarize old messages when threshold is reached
- **trim_and_summarize**: Both trim and summarize

### 3. Per-Chatbot Configuration
Memory management can be configured:
- Via environment variables (global defaults)
- Via API parameters (per-request)
- Via code (per-chatbot instance)

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Memory Management Configuration
DEFAULT_MEMORY_STRATEGY=none  # Options: none, trim, summarize, trim_and_summarize
MEMORY_TRIM_KEEP_MESSAGES=10  # Number of recent messages to keep when trimming
MEMORY_SUMMARIZE_THRESHOLD=20  # Message count threshold for summarization
```

### Example Configurations

#### HR Chatbot with Trim Strategy
```python
from app.chatbot.hr_chatbot import create_hr_chatbot
from app.chatbot.memory_config import MemoryConfig, MemoryStrategy

chatbot = create_hr_chatbot(
    memory_strategy="trim",
    trim_keep_messages=15
)
```

#### HR Chatbot with Summarize Strategy
```python
chatbot = create_hr_chatbot(
    memory_strategy="summarize",
    summarize_threshold=25
)
```

#### HR Chatbot with Both Strategies
```python
chatbot = create_hr_chatbot(
    memory_strategy="trim_and_summarize",
    trim_keep_messages=10,
    summarize_threshold=30
)
```

## API Usage

### Basic Chat (Uses Default Memory Strategy)
```bash
POST /api/v1/chat/
{
    "message": "What is the vacation policy?",
    "session_id": "user-123-session-1",
    "user_id": "user-123"
}
```

### Custom Memory Strategy per Request
```bash
POST /api/v1/chat/custom
{
    "message": "What is the vacation policy?",
    "session_id": "user-123-session-1",
    "user_id": "user-123",
    "memory_strategy": "trim",
    "trim_keep_messages": 15,
    "summarize_threshold": 25
}
```

## Architecture

### Components

1. **CheckpointerManager** (`app/chatbot/checkpointer_manager.py`)
   - Manages Redis checkpointer instance
   - Provides thread_id-based configuration
   - Singleton pattern for efficient resource usage

2. **MemoryConfig** (`app/chatbot/memory_config.py`)
   - Defines memory management strategies
   - Configurable per chatbot type
   - Supports environment variable overrides

3. **MemoryManager** (`app/chatbot/memory_manager.py`)
   - Implements trim and summarize operations
   - Processes messages based on strategy
   - Uses LLM for summarization when needed

4. **Updated ChatbotAgent** (`app/chatbot/chatbot.py`)
   - Uses checkpointer instead of chat_history parameter
   - Accepts thread_id for conversation context
   - Integrates memory management

## How It Works

### Conversation Flow

1. **User sends message** with `session_id` (used as `thread_id`)
2. **Checkpointer retrieves** conversation history from Redis
3. **Memory manager processes** messages if strategy is enabled
4. **Agent processes** the message with full context
5. **Response is saved** to checkpointer automatically
6. **Memory management** may trim/summarize if thresholds are met

### Memory Strategies

#### Trim Strategy
- Keeps only the most recent N messages (configurable)
- Removes older messages to prevent context overflow
- Fast and efficient

#### Summarize Strategy
- When message count exceeds threshold, summarizes old messages
- Keeps recent messages intact
- Uses LLM to create concise summary
- Preserves important context while reducing token usage

#### Trim and Summarize Strategy
- Combines both approaches
- Summarizes old messages and keeps recent ones
- Optimal balance between context preservation and efficiency

## Migration from Session Manager

### Before (Session Manager)
```python
session = session_manager.get_or_create_session(session_id="123")
response = session.agent.chat(
    query="Hello",
    chat_history=session.get_chat_history(format_for_agent=True)
)
```

### After (Checkpointer)
```python
chatbot = get_hr_chatbot()
response = chatbot.chat(
    query="Hello",
    thread_id="123"  # session_id is now thread_id
)
```

## Benefits

1. **Persistent Memory**: Redis storage survives server restarts
2. **Scalability**: Redis can handle high concurrency
3. **Flexibility**: Configurable memory strategies per chatbot
4. **Efficiency**: Trim/summarize reduces token usage and costs
5. **Separation of Concerns**: Memory management is separate from agent logic

## Dependencies

New dependencies added:
- `langgraph-checkpoint-redis`: Redis checkpointer for LangGraph
- `redis`: Redis Python client

Install with:
```bash
pip install -r requirements.txt
```

## Redis Setup

### Local Development
```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or install locally
# macOS: brew install redis && brew services start redis
# Linux: sudo apt-get install redis-server && sudo systemctl start redis
```

### Production
- Use managed Redis service (AWS ElastiCache, Redis Cloud, etc.)
- Configure `REDIS_URL` environment variable
- Set up proper authentication and security

## Troubleshooting

### Redis Connection Issues
- Check Redis is running: `redis-cli ping`
- Verify `REDIS_URL` is correct
- System falls back to in-memory checkpointer if Redis unavailable

### Memory Management Not Working
- Check `memory_strategy` is not "none"
- Verify thresholds are configured correctly
- Check logs for memory manager processing

### Session Not Persisting
- Verify Redis is accessible
- Check thread_id is consistent across requests
- Ensure checkpointer is initialized correctly

## Future Enhancements

- [ ] Advanced summarization with topic extraction
- [ ] Configurable TTL for Redis keys
- [ ] Memory analytics and monitoring
- [ ] Custom memory strategies per user/chatbot
- [ ] Integration with long-term memory stores

