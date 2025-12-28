# Production Deployment Guide

This guide covers best practices for deploying the RAG Chatbot to production.

## 🏗️ Architecture Overview

The application has been refactored to use Object-Oriented Programming (OOP) principles with the following key components:

### Core Classes

1. **`ChatbotAgent`** - Base class for generic chatbot functionality
2. **`HRChatbot`** - Extends `ChatbotAgent` with HR-specific prompts and tools
3. **`ChatbotSession`** - Manages individual user chat sessions
4. **`ChatbotSessionManager`** - Thread-safe manager for multiple concurrent sessions

### Key Features

- ✅ **Session Management**: Automatic session creation and management
- ✅ **Thread Safety**: Lock-based synchronization for concurrent access
- ✅ **Rate Limiting**: Built-in rate limiting middleware
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Configuration**: Environment-based configuration management

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Project Information
PROJECT_NAME="RAG Chatbot"
PROJECT_VERSION="1.0.0"

# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Session Management
MAX_CONCURRENT_SESSIONS=1000  # Optional: limit concurrent sessions
SESSION_TIMEOUT_HOURS=24      # Session expiration time

# Security
SECRET_KEY=your-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# LLM Configuration
CHAT_MODEL=gpt-4
CHAT_MODEL_TEMPERATURE=0.7
CHAT_MODEL_MAX_TOKENS=2000

# API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key  # Optional
```

## 🚀 Deployment

### 1. Using Docker (Recommended)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Using Gunicorn (Production WSGI Server)

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Using Systemd Service

Create `/etc/systemd/system/rag-chatbot.service`:

```ini
[Unit]
Description=RAG Chatbot API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/rag_chatbot
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔒 Security Best Practices

### 1. API Keys and Secrets
- Never commit API keys to version control
- Use environment variables or secret management services (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly

### 2. CORS Configuration
- Restrict CORS origins to specific domains in production
- Never use `*` in production

### 3. Rate Limiting
- Rate limiting is enabled by default (60 requests/minute, 1000/hour)
- Adjust limits based on your needs
- Consider Redis-based rate limiting for distributed systems

### 4. Input Validation
- All inputs are validated using Pydantic models
- Sanitize user inputs before processing

### 5. Error Handling
- Never expose internal error details to clients
- Log errors securely without sensitive information

## 📊 Monitoring and Logging

### Logging Configuration

The application uses structured logging. Configure log levels:

```python
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Health Checks

Monitor the health endpoint:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/chat/health
```

### Session Statistics

Get session statistics:
```bash
curl http://localhost:8000/api/v1/chat/sessions/stats
```

## 🔄 Session Management

### Session Lifecycle

1. **Creation**: Sessions are created automatically on first request
2. **Expiration**: Sessions expire after 24 hours of inactivity (configurable)
3. **Cleanup**: Expired sessions are automatically cleaned up

### Session Limits

- Set `MAX_CONCURRENT_SESSIONS` to limit concurrent users
- Monitor session count using `/api/v1/chat/sessions/stats`

### Session Persistence

For production, consider:
- **Database Storage**: Store sessions in PostgreSQL/MongoDB
- **Redis**: Use Redis for distributed session storage
- **File Storage**: Persist sessions to disk for recovery

## 🚦 Rate Limiting

### Current Implementation
- In-memory rate limiting (single server)
- 60 requests per minute per IP
- 1000 requests per hour per IP

### For Distributed Systems
Consider using:
- **Redis**: Distributed rate limiting
- **Nginx**: Rate limiting at the reverse proxy level
- **Cloudflare**: DDoS protection and rate limiting

## 📈 Performance Optimization

### 1. Caching
- Cache vector store embeddings
- Cache frequent queries
- Use Redis for distributed caching

### 2. Database Connection Pooling
- Use connection pooling for database connections
- Configure pool size based on load

### 3. Async Operations
- Use async/await for I/O operations
- Consider async database drivers

### 4. Load Balancing
- Use multiple worker processes (Gunicorn with multiple workers)
- Use load balancer (Nginx, HAProxy) for multiple servers

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
Use tools like:
- **Locust**: Python-based load testing
- **Apache Bench**: Simple HTTP benchmarking
- **k6**: Modern load testing tool

## 🔍 Troubleshooting

### Common Issues

1. **Max Sessions Reached**
   - Increase `MAX_CONCURRENT_SESSIONS`
   - Implement session cleanup
   - Use distributed session storage

2. **Rate Limit Errors**
   - Adjust rate limits in `rate_limiter.py`
   - Implement per-user rate limiting

3. **Memory Issues**
   - Monitor session count
   - Implement session cleanup
   - Use external session storage

## 📝 API Usage Examples

### Create a Chat Session

```python
import requests

# First request creates a session automatically
response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={"message": "What is the vacation policy?"}
)

session_id = response.json()["session_id"]
```

### Continue Conversation

```python
# Use session_id to maintain context
response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={
        "message": "How many days?",
        "session_id": session_id
    }
)
```

### Delete Session

```python
requests.delete(f"http://localhost:8000/api/v1/chat/sessions/{session_id}")
```

## 🎯 Next Steps

1. **Database Integration**: Add PostgreSQL/MongoDB for session persistence
2. **Authentication**: Implement user authentication and authorization
3. **Analytics**: Add usage analytics and monitoring
4. **Caching**: Implement Redis for distributed caching
5. **CDN**: Use CDN for static assets
6. **SSL/TLS**: Enable HTTPS in production
7. **Backup**: Implement regular backups of vector store and sessions

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Production Best Practices](https://fastapi.tiangolo.com/deployment/)

