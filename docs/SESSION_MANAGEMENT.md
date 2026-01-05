# Session Management

This guide covers session management, including session lifecycle, headers, shared agent pool, and best practices.

## Overview

Sessions manage conversation context for users. Each session maintains:
- Conversation history
- Session metadata (created_at, last_activity, message_count)
- Thread ID for LangChain checkpointing

## Session Lifecycle

### 1. Creation

Sessions are created automatically on first request:

```python
# No session ID provided - creates new session
response = requests.post("/api/v1/chat/", json={"message": "Hello"})
session_id = response.json()["session_id"]
```

### 2. Usage

Continue conversation using session ID:

```python
# Use session ID in subsequent requests
response = requests.post(
    "/api/v1/chat/",
    json={"message": "Follow-up question"},
    headers={"X-Session-ID": session_id}
)
```

### 3. Expiration

Sessions expire after inactivity:
- **Default**: 24 hours
- **Configurable**: `SESSION_TIMEOUT_HOURS` environment variable
- **Automatic**: Expired sessions are cleaned up automatically

### 4. Deletion

Delete session explicitly:

```python
requests.delete(f"/api/v1/chat/sessions/{session_id}")
```

## Session Headers

### Preferred Method: Headers

```
X-Session-ID: your-session-id
X-User-ID: user-123
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "X-Session-ID: abc123" \
  -H "X-User-ID: user-123" \
  -d '{"message": "Hello"}'
```

### Fallback Method: Cookies

```
session_id=your-session-id
user_id=user-123
```

**Example**:
```python
import requests

session = requests.Session()
session.cookies.set('session_id', 'abc123')
session.cookies.set('user_id', 'user-123')
response = session.post("/api/v1/chat/", json={"message": "Hello"})
```

## Session Information

### Get Session Info

```bash
curl "http://localhost:8000/api/v1/chat/sessions/{session_id}"
```

**Response**:
```json
{
  "session_id": "abc123",
  "user_id": "user-123",
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T11:45:00Z",
  "message_count": 5,
  "thread_id": "thread-abc123"
}
```

### Session Statistics

```bash
curl "http://localhost:8000/api/v1/chat/sessions/stats"
```

**Response**:
```json
{
  "total_sessions": 150,
  "total_messages": 5000,
  "agent_pool_size": 1,
  "sessions_with_custom_agents": 2
}
```

## Shared Agent Pool

The system uses a shared agent pool for efficiency:

### How It Works

- **Default**: 1 shared agent for all sessions
- **Configurable**: Set `AGENT_POOL_SIZE` environment variable
- **Thread-Safe**: Safe for concurrent requests
- **Memory Efficient**: ~99% reduction in memory usage

### Pool Size Configuration

```bash
# Single shared agent (default, recommended)
AGENT_POOL_SIZE=1

# Multiple agents for high concurrency
AGENT_POOL_SIZE=5
```

### Pool Size Guidelines

- **1**: Best for most cases, single shared agent
- **2-5**: For moderate concurrency, reduces contention
- **10+**: For very high concurrency (may be overkill)

### Benefits

- **Memory**: ~99% reduction vs per-session agents
- **Performance**: Faster session creation (no agent initialization)
- **Scalability**: Can handle many more concurrent sessions

## Session Limits

### Maximum Sessions

```bash
MAX_CONCURRENT_SESSIONS=1000
```

**Behavior**:
- If limit reached, returns `429 Too Many Requests`
- Oldest inactive sessions are cleaned up first

### Session Timeout

```bash
SESSION_TIMEOUT_HOURS=24
```

**Behavior**:
- Sessions expire after inactivity period
- Expired sessions are cleaned up automatically
- New session created if expired session used

## Memory Persistence

### Redis Checkpointing

Conversation history is stored in Redis:
- **Key**: `thread_id` (from session)
- **Storage**: LangChain checkpoint format
- **Persistence**: Survives server restarts

### Configuration

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Memory Strategies

Configured per chatbot in YAML config:

```yaml
memory:
  strategy: "trim"  # Options: "none", "trim", "summarize", "trim_and_summarize"
  trim_keep_messages: 1
  summarize_threshold: 2
```

**Strategies**:
- **none**: Keep all messages (may hit context limits)
- **trim**: Keep only last N messages
- **summarize**: Summarize old messages when threshold reached
- **trim_and_summarize**: Combine trim and summarize

## Thread Safety

Sessions are thread-safe:
- Multiple concurrent requests handled safely
- Lock-based synchronization
- No race conditions

## Best Practices

### 1. Always Use Session ID

```python
# Good: Maintains conversation context
response = requests.post(
    "/api/v1/chat/",
    json={"message": "Follow-up"},
    headers={"X-Session-ID": session_id}
)

# Bad: Loses conversation context
response = requests.post(
    "/api/v1/chat/",
    json={"message": "Follow-up"}
)
```

### 2. Store Session ID

```python
# Store session ID after first request
session_id = response.json()["session_id"]
# Use it in subsequent requests
```

### 3. Handle Expired Sessions

```python
response = requests.post(
    "/api/v1/chat/",
    json={"message": "Hello"},
    headers={"X-Session-ID": expired_session_id}
)

if response.status_code == 404:
    # Session expired, create new one
    response = requests.post(
        "/api/v1/chat/",
        json={"message": "Hello"}
    )
    session_id = response.json()["session_id"]
```

### 4. Clean Up Sessions

```python
# Delete session when done
requests.delete(f"/api/v1/chat/sessions/{session_id}")
```

### 5. Monitor Session Count

```python
# Check session statistics
stats = requests.get("/api/v1/chat/sessions/stats").json()
print(f"Total sessions: {stats['total_sessions']}")
```

## Troubleshooting

### Session Not Persisting

**Issue**: Conversation history lost between requests

**Solutions**:
- Verify Redis is running
- Check Redis connection settings
- Ensure `thread_id` is being used correctly

### Too Many Sessions

**Issue**: `429 Too Many Requests` error

**Solutions**:
- Increase `MAX_CONCURRENT_SESSIONS`
- Implement session cleanup
- Delete old/inactive sessions

### Session Expired

**Issue**: Session not found errors

**Solutions**:
- Check `SESSION_TIMEOUT_HOURS` setting
- Implement session recreation logic
- Monitor session expiration

### Memory Issues

**Issue**: High memory usage with many sessions

**Solutions**:
- Use shared agent pool (default)
- Reduce `AGENT_POOL_SIZE` if using multiple agents
- Implement session cleanup
- Use memory strategies (trim, summarize)

## Related Documentation

- [Shared Agent Pool](../SHARED_AGENT_POOL.md) - Detailed agent pool information
- [API Usage Guide](API_USAGE.md) - API integration examples
- [Configuration Guide](CONFIGURATION.md) - Session configuration options
- [Memory Management Guide](../MEMORY_MANAGEMENT_GUIDE.md) - Memory strategies

