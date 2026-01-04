# Jobs Directory

This directory contains CLI jobs for managing the RAG chatbot system.

## Vector Store Creation

The `create_vectorstore.py` script creates and populates a ChromaDB vector store for any chatbot type by loading PDF documents, splitting them into chunks, and generating embeddings. It uses the chatbot's configuration file (`{chatbot_type}_chatbot_config.yaml`) to determine settings.

### Quick Start

**For HR chatbot:**
```bash
python jobs/create_vectorstore.py --chatbot-type hr --folder /path/to/pdfs
```

**For any chatbot type:**
```bash
python jobs/create_vectorstore.py --chatbot-type <chatbot_type> --folder /path/to/pdfs
```

### Prerequisites

1. **Environment Variables**: Set your API key
   ```bash
   # For Gemini embeddings (default)
   export GOOGLE_API_KEY="your-api-key-here"
   
   # Or for OpenAI embeddings
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Required Packages**: Ensure these are installed
   - `langchain-google-genai` (for Gemini embeddings)
   - `langchain-openai` (for OpenAI embeddings, optional)
   - `langchain-community` (for ChromaDB)

3. **PDF Files**: Have your PDF documents ready in a folder

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--chatbot-type` | **Required.** Type of chatbot (e.g., 'hr', 'support', 'default') | None (required) |
| `--folder` | Path to folder containing PDF files | `/Users/kanavkalra/Data/genAI/projects/policies` |
| `--persist-dir` | Override persist directory | From `{chatbot_type}_chatbot_config.yaml` |
| `--collection-name` | Override collection name | From `{chatbot_type}_chatbot_config.yaml` |
| `--embedding-provider` | Override embedding provider: `openai`, `google`, or `auto` | From `{chatbot_type}_chatbot_config.yaml` |
| `--embedding-model` | Override embedding model name | From `{chatbot_type}_chatbot_config.yaml` |
| `--api-key` | API key (optional, uses env vars if not provided) | From env vars |
| `--chunk-size` | Maximum chunk size in characters | `1000` |
| `--chunk-overlap` | Overlap between chunks in characters | `200` |
| `--clear-existing` | Clear existing collection before adding documents | `False` |
| `--no-recursive` | Don't search PDFs recursively in subdirectories | `False` (recursive) |
| `--no-embedding-suffix` | Disable auto-generated collection names with embedding info | `False` (enabled) |
| `--skip-if-exists` | Skip document ingestion if collection already exists | `False` |

### Usage Examples

#### 1. Basic Run for HR Chatbot
Uses configuration from `hr_chatbot_config.yaml`:
```bash
python jobs/create_vectorstore.py --chatbot-type hr --folder /path/to/pdfs
```

#### 2. Custom PDF Folder
```bash
python jobs/create_vectorstore.py --chatbot-type hr --folder /path/to/your/pdfs
```

#### 3. Override Embedding Provider
```bash
python jobs/create_vectorstore.py --chatbot-type hr --embedding-provider openai
```

#### 4. Specify Custom Embedding Model
```bash
# Google/Gemini
python jobs/create_vectorstore.py --chatbot-type hr --embedding-model models/text-embedding-004

# OpenAI
python jobs/create_vectorstore.py --chatbot-type hr --embedding-provider openai --embedding-model text-embedding-3-large
```

#### 5. Clear and Rebuild Existing Collection
```bash
python jobs/create_vectorstore.py --chatbot-type hr --clear-existing
```

#### 6. Custom Chunk Settings
```bash
python jobs/create_vectorstore.py --chatbot-type hr --chunk-size 1500 --chunk-overlap 300
```

#### 7. Complete Example
```bash
python jobs/create_vectorstore.py \
  --chatbot-type hr \
  --folder /path/to/pdfs \
  --persist-dir ./chroma_db/hr_chatbot \
  --collection-name hr_chatbot \
  --embedding-provider google \
  --embedding-model models/text-embedding-004 \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --clear-existing
```

#### 8. For Different Chatbot Type
```bash
python jobs/create_vectorstore.py --chatbot-type support --folder /path/to/pdfs
```

#### 9. Multiple Embeddings Support (NEW)
The script now automatically creates separate collections for different embedding providers/models, so you can store multiple embeddings without overwriting existing ones.

**Create embeddings with OpenAI:**
```bash
python jobs/create_vectorstore.py --chatbot-type hr --embedding-provider openai --embedding-model text-embedding-3-small
```
This creates collection: `hr_chatbot_openai_text-embedding-3-small`

**Create embeddings with Google:**
```bash
python jobs/create_vectorstore.py --chatbot-type hr --embedding-provider google --embedding-model models/text-embedding-004
```
This creates collection: `hr_chatbot_google_text-embedding-004`

**Skip if collection already exists:**
```bash
python jobs/create_vectorstore.py --chatbot-type hr --embedding-provider openai --skip-if-exists
```
This is useful when you want to avoid re-ingesting documents for an existing embedding.

**Disable auto-naming (use explicit collection name):**
```bash
python jobs/create_vectorstore.py --chatbot-type hr --collection-name my_custom_collection --no-embedding-suffix
```

**Note:** To use a specific collection, update your `{chatbot_type}_chatbot_config.yaml` to set `vector_store.collection_name` to the desired collection name.

### What the Script Does

1. **Loads PDF Documents**: Recursively searches the specified folder for PDF files
2. **Splits into Chunks**: Splits documents into smaller chunks with configurable size and overlap
3. **Generates Embeddings**: Creates embeddings using the specified provider (Google/Gemini by default)
4. **Stores in ChromaDB**: Persists the vector store to disk for later use
5. **Verifies**: Checks that documents were successfully added

### Output

The script will:
- Create/update the ChromaDB collection at the specified persist directory
- Log progress information to the console
- Display the final count of document chunks stored
- Exit with code 0 on success, 1 on failure

### Troubleshooting

**Error: "Google API key is required"**
- Make sure `GOOGLE_API_KEY` environment variable is set
- Or use `--api-key` flag to provide it directly

**Error: "OpenAI embeddings require 'langchain-openai' package"**
- Install with: `pip install langchain-openai`

**Error: "Google embeddings require 'langchain-google-genai' package"**
- Install with: `pip install langchain-google-genai`

**No PDFs found**
- Check that the `--folder` path is correct
- Ensure PDF files exist in that directory
- Remove `--no-recursive` if PDFs are in subdirectories

**Collection already exists**
- Use `--clear-existing` to rebuild from scratch
- Use `--skip-if-exists` to skip ingestion if collection already exists
- Or the script will append to existing collection (may create duplicates)

**Multiple Embeddings**
- By default, collection names include the embedding provider and model (e.g., `hr_chatbot_openai_text-embedding-3-small`)
- This allows you to store multiple embeddings without overwriting
- To use a specific collection, update `vector_store.collection_name` in your config YAML
- Use `--no-embedding-suffix` to disable auto-naming and use the base collection name from config

### Integration with Chatbots

The vector store created by this script is automatically loaded by the chatbot service when it starts. The chatbot uses the vector store configuration from `{chatbot_type}_chatbot_config.yaml`:
- `vector_store.persist_dir`: Persistence directory for ChromaDB
- `vector_store.collection_name`: ChromaDB collection name
- `vector_store.embedding_provider`: Embedding provider (`auto`, `openai`, or `google`)

Make sure the embedding provider used to create the vector store matches what the chatbot expects, or the chatbot won't be able to load it correctly.

### Multiple Embeddings Support

The script now supports storing multiple embeddings from different providers/models simultaneously. This is achieved by automatically generating collection names that include the embedding provider and model information.

**How it works:**
- Collection names are auto-generated as: `{base_name}_{provider}_{model_slug}`
- Example: `hr_chatbot_openai_text-embedding-3-small` or `hr_chatbot_google_embedding-001`
- Each embedding provider/model gets its own collection
- You can switch between embeddings by updating `vector_store.collection_name` in your config

**Benefits:**
- No need to re-run ingestion when changing embedding models
- Compare different embeddings easily
- Keep multiple embeddings available for A/B testing

**Usage:**
1. Create embeddings with different providers/models - each will create its own collection
2. Update your `{chatbot_type}_chatbot_config.yaml` to point to the desired collection
3. The chatbot will automatically use the specified collection

