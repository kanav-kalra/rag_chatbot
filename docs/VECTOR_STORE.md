# Vector Store Management

This guide covers creating, managing, and updating vector stores for chatbots.

## Overview

Vector stores are used to store document embeddings for RAG (Retrieval-Augmented Generation). The application uses ChromaDB for persistent vector storage.

## Vector Store Location

Vector stores are stored in:
```
data/vectorstores/chroma_db/{chatbot_type}/
```

Each chatbot type has its own directory and collection.

## Creating a Vector Store

### Using the Script (Recommended)

For HR chatbot:
```bash
./scripts/jobs/create_hr_vectorstore.sh --folder-path /path/to/pdfs
```

For other chatbots:
```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type support \
  --folder-path /path/to/pdfs
```

### Direct Command

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type hr \
  --folder-path /path/to/pdfs \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --recursive
```

### Command Options

- **`--chatbot-type`**: Chatbot type identifier (required)
- **`--folder`**: Path to folder containing PDFs (default: from `vector_store.ingestion.folder_path` in config)
- **`--chunk-size`**: Maximum chunk size in characters (default: from `vector_store.ingestion.chunk_size` in config, or 1000)
- **`--chunk-overlap`**: Overlap between chunks in characters (default: from `vector_store.ingestion.chunk_overlap` in config, or 200)
- **`--recursive`**: Search subdirectories (default: from `vector_store.ingestion.recursive` in config, or true)
- **`--indexing-mode`**: Indexing mode - `incremental` (default) or `full`
  - **`incremental`**: Only indexes new or changed files (efficient, skips if no changes)
  - **`full`**: Always re-indexes everything (clears existing collection)
- **`--clear-existing`**: Delete existing collection before adding documents (equivalent to `--indexing-mode full`)
- **`--skip-if-exists`**: Skip if collection already exists
- **`--embedding-provider`**: Override embedding provider (auto/openai/google)
- **`--embedding-model`**: Override embedding model

### Example

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type hr \
  --folder-path ./documents/hr_policies \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --recursive
```

## Document Processing

### Supported Formats

- **PDF**: `.pdf` files
- More formats can be added by extending the loader

### Chunking Strategy

Documents are split into chunks with:
- **Chunk Size**: Maximum characters per chunk (default: 1000)
- **Chunk Overlap**: Characters overlapping between chunks (default: 200)

**Why Overlap?**
- Prevents losing context at chunk boundaries
- Improves retrieval quality for queries spanning chunks

### Embedding Generation

Embeddings are generated using:
- **Provider**: Configured in chatbot config (`embedding_provider`)
- **Model**: Configured in chatbot config (`embedding_model`)
- **Auto-detection**: If `embedding_provider: "auto"`, detects based on LLM model

## Updating a Vector Store

### Incremental Indexing (Default)

By default, the script uses **incremental indexing** mode, which only indexes new or changed files. This is the most efficient approach for regular updates.

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type hr \
  --folder /path/to/pdfs \
  --indexing-mode incremental
```

**How it works:**
- Compares file modification times and sizes with indexed files
- Only processes new files or files that have changed
- Removes old chunks for changed files before re-indexing
- Skips indexing entirely if no changes detected
- Preserves existing embeddings for unchanged documents

**Benefits:**
- Fast updates (only processes what changed)
- Efficient resource usage
- Automatic change detection

### Full Re-indexing

To completely rebuild a vector store (clears existing and re-indexes everything):

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type hr \
  --folder /path/to/pdfs \
  --indexing-mode full
```

Or use the equivalent:

```bash
python scripts/ingestion/create_vectorstore.py \
  --chatbot-type hr \
  --folder /path/to/pdfs \
  --clear-existing
```

**Warning**: This deletes all existing documents in the collection and rebuilds from scratch.

**When to use:**
- After changing chunk size or overlap settings
- When switching embedding providers/models
- When you want to ensure a completely fresh index

## Multiple Embeddings

The system automatically supports storing multiple embeddings for the same documents:
- Different embedding providers (OpenAI, Google)
- Different embedding models
- **Collection names are automatically suffixed with provider/model information**

Example collection names (auto-generated):
- `hr_chatbot_openai_text-embedding-3-small` (when using OpenAI)
- `hr_chatbot_google_embedding-001` (when using Google)

**Key Benefit**: You can switch embedding providers in your config without recreating embeddings. Each provider/model combination gets its own collection, allowing you to test different embeddings easily.

## Vector Store Configuration

Configure in chatbot YAML config:

```yaml
vector_store:
  type: "hr"
  persist_dir: "./data/vectorstores/chroma_db/hr_chatbot"
  collection_name: "hr_chatbot"  # Base name (will be auto-suffixed)
  embedding_provider: "auto"  # or "openai", "google"
  embedding_model: ""  # Empty = use provider default
  
  # Ingestion Configuration (for create_vectorstore.py script)
  ingestion:
    folder_path: "/path/to/pdfs"  # Default folder path containing PDF files
    chunk_size: 1000  # Maximum size of chunks to return (in characters)
    chunk_overlap: 200  # Overlap in characters between chunks
    recursive: true  # If true, search for PDFs recursively in subdirectories
    indexing_mode: "incremental"  # "incremental" (default) or "full"
```

### Configuration Options

- **`type`**: Must match chatbot type
- **`persist_dir`**: Where ChromaDB stores data
- **`collection_name`**: **Base** collection name (automatically suffixed with `_{provider}_{model}`)
- **`embedding_provider`**: `auto`, `openai`, or `google` (auto-detects based on LLM model if set to "auto")
- **`embedding_model`**: Specific model or empty for provider default

**Note**: The `collection_name` in config is treated as a base name. The system automatically appends the embedding provider and model to create the actual collection name. This allows you to switch providers by simply changing `embedding_provider` in the config - no need to recreate embeddings!

## Querying the Vector Store

The vector store is queried automatically by the retrieval tool:

```python
# This happens automatically when chatbot processes a query
retrieval_service.retrieve_documents(query="What is the vacation policy?")
```

The retrieval service:
1. Converts query to embedding
2. Searches vector store for similar documents
3. Returns top K most relevant chunks (default: 5)

## Monitoring

### Check Vector Store Status

```python
from src.infrastructure.vectorstore.manager import get_vector_store

vector_store = get_vector_store("hr")
# Check if collection exists and has documents
```

### View Collection Info

ChromaDB stores metadata about collections. You can inspect:
- Number of documents
- Collection name
- Embedding dimensions

## Best Practices

### Chunk Size

- **Small chunks (500-1000)**: Better for precise retrieval, more chunks
- **Large chunks (1500-2000)**: Better for context, fewer chunks
- **Recommended**: 1000-1500 characters

### Chunk Overlap

- **Small overlap (100-200)**: Faster processing, may lose context
- **Large overlap (300-500)**: Better context, slower processing
- **Recommended**: 200-300 characters

### Document Organization

- Organize documents by topic in subdirectories
- Use descriptive filenames
- Include metadata in filenames if needed

### Regular Updates

- Use incremental indexing (default) for regular updates - it automatically detects and indexes only changed files
- Run full re-indexing when changing chunk settings or embedding providers
- Monitor document count to ensure updates are working
- Check retrieval quality regularly

## Troubleshooting

### Vector Store Not Found

**Error**: `Collection not found`

**Solutions**:
- Ensure vector store was created
- Check `persist_dir` path in config
- Verify collection name matches

### Empty Results

**Issue**: No documents retrieved

**Solutions**:
- Check if documents were added successfully
- Verify embedding provider/model
- Check query relevance

### Slow Retrieval

**Issue**: Retrieval takes too long

**Solutions**:
- Reduce chunk size
- Use smaller embedding models
- Optimize document count

### Memory Issues

**Issue**: High memory usage

**Solutions**:
- Reduce chunk size
- Limit document count
- Use smaller embedding models

## Related Documentation

- [Creating a New Chatbot](CREATING_NEW_CHATBOT.md) - Creating vector stores for new chatbots
- [Configuration Guide](CONFIGURATION.md) - Vector store configuration
- [HR Chatbot Flow](HR_CHATBOT_FLOW.md) - How retrieval works in the flow

