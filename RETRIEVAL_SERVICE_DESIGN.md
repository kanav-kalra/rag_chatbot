# Retrieval Service Design

## Overview

The `RetrievalService` class provides a generic, domain-agnostic interface for document retrieval from vector stores. It follows the **dependency injection** pattern and has **no knowledge** of chatbot types, HR, or any domain-specific concepts.

## Architecture

### Separation of Concerns

```
┌─────────────────────────────────┐
│   Chatbot Classes (hr_chatbot)  │  ← Knows about chatbot types
│   - Gets vector store            │
│   - Creates RetrievalService    │
│   - Creates tools               │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   RetrievalService              │  ← Generic, no domain knowledge
│   - Takes VectorStore in ctor    │
│   - Provides retrieval methods   │
│   - Creates LangChain tools      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   VectorStore (from LangChain)  │  ← Generic vector store interface
└─────────────────────────────────┘
```

### What is `vector_store_manager.py`?

The `vector_store_manager.py` is an **infrastructure layer** that:

1. **Manages Vector Store Lifecycle**
   - Loads vector stores from disk (ChromaDB)
   - Caches loaded vector stores in memory
   - Handles thread-safe access to cached stores

2. **Maps Chatbot Types to Vector Stores**
   - Each chatbot type (e.g., "hr", "support") has its own vector store
   - Manages configuration per chatbot type:
     - `persist_dir`: Where the ChromaDB data is stored
     - `collection_name`: ChromaDB collection name
     - `store_type`: Type of vector store ("chroma", "memory", etc.)

3. **Provides `get_vector_store(chatbot_type)`**
   - This is the **bridge** between chatbot types and their vector stores
   - Returns a `VectorStore` instance for the given chatbot type
   - Handles lazy loading and caching

**Key Point**: `vector_store_manager.py` knows about chatbot types because it's responsible for the infrastructure concern of "which vector store belongs to which chatbot type". This is appropriate because it's an infrastructure/configuration layer.

## Design Principles

### 1. Dependency Injection

The `RetrievalService` class takes a `VectorStore` in its constructor:

```python
vector_store = get_vector_store("hr")  # Infrastructure layer handles chatbot type
retrieval_service = RetrievalService(vector_store)  # Service doesn't know about "hr"
```

### 2. Single Responsibility

- **RetrievalService**: Only responsible for retrieving documents from a vector store
- **vector_store_manager**: Only responsible for loading and managing vector stores
- **Chatbot classes**: Responsible for wiring everything together

### 3. No Domain Knowledge in RetrievalService

The `RetrievalService` class:
- ✅ Works with any `VectorStore`
- ✅ Has no references to "hr", "chatbot_type", or domain concepts
- ✅ Can be reused for any chatbot or use case
- ✅ Is easily testable (just inject a mock VectorStore)

## Usage Example

### In Chatbot Classes (e.g., `hr_chatbot.py`)

```python
from app.services.retrieval.retrieval_service import RetrievalService
from app.infra.vectorstore import get_vector_store

class HRChatbot(ChatbotAgent):
    def __init__(self, ...):
        # 1. Get vector store for this chatbot type
        vector_store = get_vector_store("hr")
        
        # 2. Create retrieval service with the vector store
        retrieval_service = RetrievalService(vector_store)
        
        # 3. Create tool from the service
        retrieve_documents_tool = retrieval_service.create_tool()
        
        # 4. Use the tool in the chatbot
        tools = [retrieve_documents_tool]
        # ... rest of initialization
```

### Direct Usage

```python
from app.services.retrieval.retrieval_service import RetrievalService
from app.infra.vectorstore import get_vector_store

# Get vector store
vector_store = get_vector_store("hr")

# Create service
service = RetrievalService(vector_store)

# Retrieve documents
content, artifact = service.retrieve(query="What is the notice period?", k=4)

# Or get with scores
results = service.retrieve_with_scores(query="...", k=4)

# Or get as string
content_string = service.retrieve_as_string(query="...", k=4)
```

## API Reference

### `RetrievalService` Class

#### Constructor

```python
RetrievalService(vector_store: VectorStore)
```

- `vector_store`: A LangChain `VectorStore` instance (required)

#### Methods

##### `retrieve(query: str, k: int = 4) -> Tuple[str, List[Dict[str, Any]]]`

Retrieve documents and return formatted content and artifact.

- Returns: `(content_string, artifact_list)`
- `content_string`: Formatted string with all documents
- `artifact_list`: List of dicts with `rank`, `content`, `metadata`

##### `retrieve_with_scores(query: str, k: int = 4) -> List[Dict[str, Any]]`

Retrieve documents with similarity scores.

- Returns: List of dicts with `rank`, `content`, `metadata`, `score`

##### `retrieve_as_string(query: str, k: int = 4, separator: str = "\n\n---\n\n") -> str`

Retrieve documents as a formatted string.

- Returns: Formatted string with all documents

##### `create_tool(tool_name: Optional[str] = None, description: Optional[str] = None) -> BaseTool`

Create a LangChain tool from this service instance.

- Returns: A LangChain tool that can be used by agents
- The tool uses `response_format="content_and_artifact"`

## Migration Notes

### Backward Compatibility

The old function-based API is still available but deprecated:

- `retrieve_documents()` - Deprecated, use `RetrievalService.create_tool()` instead
- `retrieve_documents_generic()` - Deprecated, use `RetrievalService` class instead

These will be removed in a future version.

### Migration Steps

1. ✅ `RetrievalService` class created
2. ✅ `hr_chatbot.py` updated to use new class
3. ⏳ Update any other code that uses old functions
4. ⏳ Remove deprecated functions after migration complete

## Benefits

1. **Testability**: Easy to test with mock vector stores
2. **Reusability**: Same service works for any chatbot type
3. **Separation of Concerns**: Service doesn't know about chatbot types
4. **Flexibility**: Easy to swap vector stores or add new retrieval strategies
5. **Maintainability**: Clear boundaries between layers

