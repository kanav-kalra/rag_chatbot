# Jobs Directory

This directory contains CLI jobs for managing the RAG chatbot system.

## HR Vector Store Creation

The `create_hr_vectorstore.py` script creates and populates a ChromaDB vector store for the HR chatbot by loading PDF documents, splitting them into chunks, and generating embeddings.

### Quick Start

**Using the shell script (recommended):**
```bash
./jobs/create_hr_vectorstore.sh
```

**Or directly with Python:**
```bash
python jobs/create_hr_vectorstore.py
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
| `--folder` | Path to folder containing PDF files | `/Users/kanavkalra/Data/genAI/projects/policies` |
| `--persist-dir` | Directory where ChromaDB persists data | `./chroma_db/hr_chatbot` |
| `--collection-name` | ChromaDB collection name | `hr_chatbot` |
| `--embedding-provider` | Embedding provider: `openai` or `google` | `google` |
| `--embedding-model` | Embedding model name (optional, uses provider default) | Provider default |
| `--api-key` | API key (optional, uses env vars if not provided) | From env vars |
| `--chunk-size` | Maximum chunk size in characters | `1000` |
| `--chunk-overlap` | Overlap between chunks in characters | `200` |
| `--clear-existing` | Clear existing collection before adding documents | `False` |
| `--no-recursive` | Don't search PDFs recursively in subdirectories | `False` (recursive) |

### Usage Examples

#### 1. Basic Run (with defaults)
Uses Gemini embeddings, default PDF folder, and default settings:
```bash
python jobs/create_hr_vectorstore.py
```

#### 2. Custom PDF Folder
```bash
python jobs/create_hr_vectorstore.py --folder /path/to/your/pdfs
```

#### 3. Use OpenAI Embeddings
```bash
python jobs/create_hr_vectorstore.py --embedding-provider openai
```

#### 4. Specify Custom Embedding Model
```bash
# Google/Gemini
python jobs/create_hr_vectorstore.py --embedding-model models/text-embedding-004

# OpenAI
python jobs/create_hr_vectorstore.py --embedding-provider openai --embedding-model text-embedding-3-large
```

#### 5. Clear and Rebuild Existing Collection
```bash
python jobs/create_hr_vectorstore.py --clear-existing
```

#### 6. Custom Chunk Settings
```bash
python jobs/create_hr_vectorstore.py --chunk-size 1500 --chunk-overlap 300
```

#### 7. Complete Example
```bash
python jobs/create_hr_vectorstore.py \
  --folder /path/to/pdfs \
  --persist-dir ./chroma_db/hr_chatbot \
  --collection-name hr_chatbot \
  --embedding-provider google \
  --embedding-model models/text-embedding-004 \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --clear-existing
```

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
- Or the script will append to existing collection (may create duplicates)

### Integration with HR Chatbot

The vector store created by this script is automatically loaded by the HR chatbot service when it starts. The chatbot uses the vector store configuration from `app/core/config.py`:
- `HR_CHROMA_PERSIST_DIR`: Default `./chroma_db/hr_chatbot`
- `HR_CHROMA_COLLECTION_NAME`: Default `hr_chatbot`
- `HR_EMBEDDING_PROVIDER`: Default `auto` (auto-detects based on chat model)

Make sure the embedding provider used to create the vector store matches what the chatbot expects, or the chatbot won't be able to load it correctly.

