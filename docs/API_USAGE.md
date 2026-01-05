# API Usage Guide

This guide covers how to use the RAG Chatbot API, including endpoints, request/response formats, and code examples.

## Base URL

```
http://localhost:8000/api/v1/chat
```

## Authentication

Currently, the API does not require authentication. Session management is handled via headers or cookies.

## Chat Endpoint

### POST `/api/v1/chat/`

Send a message to the HR chatbot and receive a response.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your-session-id" \
  -d '{
    "message": "What is the vacation policy?"
  }'
```

**Request Body**:
```json
{
  "message": "What is the vacation policy?"
}
```

**Request Headers** (optional):
- `X-Session-ID`: Session identifier (if continuing a conversation)
- `X-User-ID`: User identifier (optional)

**Response**:
```json
{
  "response": "The vacation policy allows employees to accrue 1.25 days per month...",
  "session_id": "abc123",
  "model_used": "gemini-2.5-flash",
  "message_count": 5
}
```

**Response Fields**:
- `response`: The chatbot's response text
- `session_id`: Session identifier (use in next request)
- `model_used`: LLM model that generated the response
- `message_count`: Number of messages in this session

**Status Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid request body
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Session Management

### Get Session Info

**GET** `/api/v1/chat/sessions/{session_id}`

Get information about a session.

**Request**:
```bash
curl "http://localhost:8000/api/v1/chat/sessions/abc123"
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

### Delete Session

**DELETE** `/api/v1/chat/sessions/{session_id}`

Delete a session and its conversation history.

**Request**:
```bash
curl -X DELETE "http://localhost:8000/api/v1/chat/sessions/abc123"
```

**Response**:
```json
{
  "message": "Session deleted successfully"
}
```

### Get Session Statistics

**GET** `/api/v1/chat/sessions/stats`

Get statistics about all sessions.

**Request**:
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

## Python Client Examples

### Basic Usage

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/chat"

# First request - creates session automatically
response = requests.post(
    f"{BASE_URL}/",
    json={"message": "What is the vacation policy?"}
)
data = response.json()
session_id = data["session_id"]
print(f"Response: {data['response']}")

# Continue conversation
response = requests.post(
    f"{BASE_URL}/",
    json={"message": "How many days?"},
    headers={"X-Session-ID": session_id}
)
print(f"Response: {response.json()['response']}")
```

### Session Management

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/chat"
session_id = None

def chat(message):
    global session_id
    
    headers = {}
    if session_id:
        headers["X-Session-ID"] = session_id
    
    response = requests.post(
        f"{BASE_URL}/",
        json={"message": message},
        headers=headers
    )
    
    data = response.json()
    session_id = data["session_id"]
    return data["response"]

# Use it
response1 = chat("What is the vacation policy?")
print(response1)

response2 = chat("How many days?")
print(response2)
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

BASE_URL = "http://localhost:8000/api/v1/chat"

def chat_safe(message, session_id=None):
    try:
        headers = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        
        response = requests.post(
            f"{BASE_URL}/",
            json={"message": message},
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Rate limit exceeded. Please wait.")
        elif e.response.status_code == 500:
            print("Server error. Please try again later.")
        else:
            print(f"HTTP error: {e}")
        return None
    except RequestException as e:
        print(f"Request error: {e}")
        return None

# Use it
result = chat_safe("What is the vacation policy?")
if result:
    print(result["response"])
```

### Class-Based Client

```python
import requests
from typing import Optional

class ChatbotClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1/chat"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    def chat(self, message: str) -> dict:
        """Send a message and get response."""
        headers = {}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        
        response = requests.post(
            f"{self.base_url}/",
            json={"message": message},
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        self.session_id = data["session_id"]
        return data
    
    def get_session_info(self) -> dict:
        """Get current session information."""
        if not self.session_id:
            raise ValueError("No active session")
        
        response = requests.get(
            f"{self.base_url}/sessions/{self.session_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def delete_session(self):
        """Delete current session."""
        if not self.session_id:
            return
        
        requests.delete(f"{self.base_url}/sessions/{self.session_id}")
        self.session_id = None

# Use it
client = ChatbotClient()
response = client.chat("What is the vacation policy?")
print(response["response"])

# Continue conversation
response = client.chat("How many days?")
print(response["response"])

# Get session info
info = client.get_session_info()
print(f"Messages: {info['message_count']}")

# Delete session
client.delete_session()
```

## JavaScript/TypeScript Examples

### Basic Usage

```javascript
const BASE_URL = 'http://localhost:8000/api/v1/chat';
let sessionId = null;

async function chat(message) {
    const headers = {
        'Content-Type': 'application/json',
    };
    
    if (sessionId) {
        headers['X-Session-ID'] = sessionId;
    }
    
    const response = await fetch(`${BASE_URL}/`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ message })
    });
    
    const data = await response.json();
    sessionId = data.session_id;
    return data.response;
}

// Use it
const response = await chat('What is the vacation policy?');
console.log(response);
```

### React Hook

```javascript
import { useState, useCallback } from 'react';

function useChatbot() {
    const [sessionId, setSessionId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const chat = useCallback(async (message) => {
        setLoading(true);
        setError(null);
        
        try {
            const headers = {
                'Content-Type': 'application/json',
            };
            
            if (sessionId) {
                headers['X-Session-ID'] = sessionId;
            }
            
            const response = await fetch('http://localhost:8000/api/v1/chat/', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            setSessionId(data.session_id);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [sessionId]);
    
    return { chat, loading, error, sessionId };
}

// Use it in component
function ChatComponent() {
    const { chat, loading, error } = useChatbot();
    const [response, setResponse] = useState('');
    
    const handleSend = async (message) => {
        try {
            const data = await chat(message);
            setResponse(data.response);
        } catch (err) {
            console.error(err);
        }
    };
    
    return (
        <div>
            <button onClick={() => handleSend('Hello')} disabled={loading}>
                Send
            </button>
            {error && <p>Error: {error}</p>}
            {response && <p>{response}</p>}
        </div>
    );
}
```

## Rate Limiting

The API has rate limiting enabled:
- **60 requests per minute** per IP
- **1000 requests per hour** per IP

**Rate Limit Headers**:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets

**Rate Limit Exceeded Response**:
```json
{
  "detail": "Rate limit exceeded: 60 requests per minute"
}
```

Status code: `429 Too Many Requests`

## Session Headers

### Preferred Method: Headers

```
X-Session-ID: your-session-id
X-User-ID: user-123
```

### Fallback Method: Cookies

```
session_id=your-session-id
user_id=user-123
```

## WebSocket Support (Future)

WebSocket support for streaming responses is planned for future releases.

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Message cannot be empty"
}
```

### 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded: 60 requests per minute"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to get response from chatbot: [error message]"
}
```

## Best Practices

1. **Session Management**: Always use `X-Session-ID` header to maintain conversation context
2. **Error Handling**: Implement proper error handling for all requests
3. **Rate Limiting**: Respect rate limits and implement retry logic with backoff
4. **Timeout**: Set appropriate timeouts (30 seconds recommended)
5. **Logging**: Log requests and responses for debugging

## Related Documentation

- [Session Management](SESSION_MANAGEMENT.md) - Session details
- [Configuration Guide](CONFIGURATION.md) - Configuration options
- [HR Chatbot Flow](HR_CHATBOT_FLOW.md) - Request processing flow

