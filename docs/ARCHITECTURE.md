# Architecture Guide

This document describes the system architecture, layer responsibilities, and design principles of the RAG Chatbot application.

## Architecture Overview

The project follows **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                    ┌──────────────┐          │
│  │   FastAPI    │                    │  Streamlit   │          │
│  │   (REST API) │                    │     (UI)     │          │
│  └──────┬───────┘                    └──────┬───────┘          │
└─────────┼────────────────────────────────────┼──────────────────┘
          │                                    │
          └────────────────┬───────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                   Application Layer                             │
├──────────────────────────┼──────────────────────────────────────┤
│  ┌───────────────────────▼───────────────────────┐             │
│  │         Chatbot Use Cases                      │             │
│  │  - Agent Pool Management                       │             │
│  │  - Graph Factory                               │             │
│  └───────────────────────┬───────────────────────┘             │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────┐             │
│  │         Ingestion Use Cases                    │             │
│  │  - Document Loading                            │             │
│  │  - Document Chunking                           │             │
│  │  - Embedding Generation                        │             │
│  └────────────────────────────────────────────────┘             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                      Domain Layer                                │
├──────────────────────────┼──────────────────────────────────────┤
│  ┌───────────────────────▼───────────────────────┐             │
│  │              Chatbot Domain                     │             │
│  │  ┌─────────────────────────────────────────┐  │             │
│  │  │  ChatbotAgent (Base Class)              │  │             │
│  │  │  - HRChatbot (Implementation)            │  │             │
│  │  │  - ConfigManager                         │  │             │
│  │  │  - ToolFactory                           │  │             │
│  │  │  - PromptBuilder                          │  │             │
│  │  └─────────────────────────────────────────┘  │             │
│  └────────────────────────────────────────────────┘             │
│  ┌────────────────────────────────────────────────┐             │
│  │         Memory Domain                         │             │
│  │  - MemoryManager                              │             │
│  │  - MemoryConfig                                │             │
│  └────────────────────────────────────────────────┘             │
│  ┌────────────────────────────────────────────────┐             │
│  │         Retrieval Domain                      │             │
│  │  - RetrievalService                            │             │
│  └────────────────────────────────────────────────┘             │
│  ┌────────────────────────────────────────────────┐             │
│  │         Session Domain                        │             │
│  │  - ChatbotSession                              │             │
│  │  - ChatbotSessionManager                       │             │
│  └────────────────────────────────────────────────┘             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                  Infrastructure Layer                            │
├──────────────────────────┼──────────────────────────────────────┤
│  ┌───────────────────────▼───────────────────────┐             │
│  │              LLM Providers                     │             │
│  │  - OpenAI, Anthropic, Google, Ollama          │             │
│  └────────────────────────────────────────────────┘             │
│  ┌────────────────────────────────────────────────┐             │
│  │           Vector Store                         │             │
│  │  - ChromaDB Manager                            │             │
│  │  - Embedding Providers                         │             │
│  └────────────────────────────────────────────────┘             │
│  ┌────────────────────────────────────────────────┐             │
│  │              Storage                           │             │
│  │  - Redis Checkpointing                         │             │
│  │  - File Storage                                │             │
│  └────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Domain Layer (`src/domain/`)

**Purpose**: Core business logic, entities, and domain models. No dependencies on external frameworks.

**Components**:
- **Chatbot Domain** (`chatbot/`): Chatbot agents, configuration, tools, prompts
- **Memory Domain** (`memory/`): Memory management strategies
- **Retrieval Domain** (`retrieval/`): Document retrieval service
- **Session Domain** (`session/`): Session management

**Key Classes**:
- `ChatbotAgent`: Base class for all chatbots
- `HRChatbot`: HR-specific chatbot implementation
- `MemoryManager`: Manages conversation memory
- `RetrievalService`: Handles document retrieval
- `ChatbotSession`: Represents a user session

### Application Layer (`src/application/`)

**Purpose**: Use cases and orchestration logic. Coordinates domain objects.

**Components**:
- **Chatbot Use Cases** (`chatbot/`): Agent pool management, graph factory
- **Ingestion Use Cases** (`ingestion/`): Document loading, chunking, embedding

**Key Classes**:
- `AgentPool`: Manages shared agent instances
- `DocumentLoader`: Loads documents from various sources
- `DocumentChunker`: Splits documents into chunks
- `Embedder`: Generates embeddings for documents

### Infrastructure Layer (`src/infrastructure/`)

**Purpose**: External services and implementations.

**Components**:
- **LLM** (`llm/`): LLM provider integrations (OpenAI, Anthropic, Google, Ollama)
- **Vector Store** (`vectorstore/`): ChromaDB and embedding providers
- **Storage** (`storage/`): Redis checkpointing, file storage

**Key Classes**:
- `LLMManager`: Manages LLM instances
- `VectorStoreManager`: Manages vector store instances
- `Checkpointer`: Handles Redis checkpointing

### API Layer (`src/api/`)

**Purpose**: FastAPI routes and middleware.

**Components**:
- **Middleware** (`middleware/`): Rate limiting, CORS, etc.
- **Routes** (`v1/routes/`): API endpoints

**Key Files**:
- `chat.py`: Chat endpoints
- `rate_limiter.py`: Rate limiting middleware

### UI Layer (`src/ui/`)

**Purpose**: Streamlit application for testing and demonstration.

**Components**:
- **Pages** (`pages/`): Streamlit pages (chatbot, dashboard, API explorer)

### Shared (`src/shared/`)

**Purpose**: Common utilities and configuration.

**Components**:
- **Config** (`config/`): Application settings, logging
- **Dependencies** (`dependencies/`): Dependency injection
- **Memory** (`memory/`): Memory configuration

## Project Structure

```
rag_chatbot/
├── src/
│   ├── main.py                   # Application entry point
│   ├── domain/                   # Domain layer
│   │   ├── chatbot/
│   │   │   ├── core/            # Core chatbot framework
│   │   │   │   ├── chatbot_agent.py
│   │   │   │   ├── config.py
│   │   │   │   ├── prompts.py
│   │   │   │   └── tools.py
│   │   │   └── hr_chatbot.py
│   │   ├── memory/
│   │   ├── retrieval/
│   │   └── session/
│   ├── application/              # Application layer
│   │   ├── chatbot/
│   │   └── ingestion/
│   ├── infrastructure/           # Infrastructure layer
│   │   ├── llm/
│   │   ├── vectorstore/
│   │   └── storage/
│   ├── api/                      # API layer
│   │   ├── middleware/
│   │   └── v1/
│   ├── ui/                       # UI layer
│   │   └── pages/
│   └── shared/                  # Shared utilities
│       ├── config/
│       ├── memory/
│       └── dependencies/
├── scripts/                      # Scripts and tools
├── config/                       # Configuration files
├── data/                         # Data directories
└── docs/                         # Documentation
```

## Design Principles

### Clean Architecture

- **Dependency Rule**: Dependencies point inward. Outer layers depend on inner layers, not vice versa.
- **Independence**: Business logic is independent of frameworks, UI, and databases.
- **Testability**: Business logic can be tested without external dependencies.

### SOLID Principles

- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subclasses can replace base classes
- **Interface Segregation**: Clients shouldn't depend on unused interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

### Key Patterns

- **Dependency Injection**: Dependencies are injected, not created internally
- **Factory Pattern**: Agent pool uses factory pattern for agent creation
- **Strategy Pattern**: Memory management uses strategy pattern
- **Repository Pattern**: Vector store abstraction follows repository pattern

## Data Flow

1. **Request** → FastAPI endpoint
2. **Session** → Session manager retrieves/creates session
3. **Agent** → Agent pool provides chatbot instance
4. **Query** → Chatbot processes query
5. **Retrieval** → Retrieval service searches vector store
6. **LLM** → LLM generates response with context
7. **Memory** → Conversation saved to Redis checkpoint
8. **Response** → Formatted response returned to user

See [HR Chatbot Flow](HR_CHATBOT_FLOW.md) for detailed sequence diagram.

## Extension Points

### Creating New Chatbots

1. Create YAML config file in `config/chatbot/`
2. Create prompts file in `config/chatbot/prompts/`
3. Subclass `ChatbotAgent` with minimal implementation
4. Create vector store using ingestion scripts

See [Creating a New Chatbot](CREATING_NEW_CHATBOT.md) for detailed guide.

### Adding New LLM Providers

1. Extend `LLMManager` to support new provider
2. Add provider-specific configuration
3. Update environment variables

### Adding New Vector Stores

1. Implement vector store interface
2. Update `VectorStoreManager`
3. Add configuration options

## Performance Considerations

- **Shared Agent Pool**: Reduces memory usage by ~99%
- **Redis Checkpointing**: Fast session persistence
- **Lazy Loading**: Vector stores loaded on demand
- **Caching**: Configuration and vector stores are cached

## Security

- **Rate Limiting**: API endpoints have rate limiting
- **Input Validation**: Pydantic models validate all inputs
- **Session Isolation**: Each session is isolated
- **API Key Management**: Keys stored in environment variables

## Related Documentation

- [HR Chatbot Flow](HR_CHATBOT_FLOW.md) - Request processing flow
- [Creating a New Chatbot](CREATING_NEW_CHATBOT.md) - Extension guide
- [Configuration Guide](CONFIGURATION.md) - Configuration details
- [Session Management](SESSION_MANAGEMENT.md) - Session architecture

