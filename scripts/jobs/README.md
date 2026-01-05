# HR Chatbot Vector Store Creation Script

## Overview

`create_hr_vectorstore.sh` is a shell script wrapper that simplifies the creation of vector stores for the HR chatbot. It automatically sets up the correct working directory and passes the `--chatbot-type hr` parameter to the underlying Python ingestion script.

## Prerequisites

- Python 3.x installed
- All project dependencies installed (see `requirements.txt`)
- Access to the folder containing PDF documents to ingest
- Required environment variables set (API keys for embedding providers if needed)

## Usage

### Basic Usage

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder <path_to_pdf_folder>
```

### Example

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /Users/kanavkalra/Data/genAI/projects/policies
```

## Available Options

All arguments passed to the shell script are forwarded to the underlying Python script. The script automatically adds `--chatbot-type hr`, so you don't need to specify it.

### Required Arguments

- `--folder <path>`: Path to the folder containing PDF files to ingest

### Optional Arguments

- `--persist-dir <path>`: Override the persist directory (default: from `hr_chatbot_config.yaml`)
- `--collection-name <name>`: Override the collection name (default: from `hr_chatbot_config.yaml`)
- `--chunk-size <size>`: Chunk size for text splitting (default: 1000)
- `--chunk-overlap <overlap>`: Chunk overlap for text splitting (default: 200)
- `--embedding-provider <provider>`: Override embedding provider (`openai`, `google`, or `auto`)
- `--embedding-model <model>`: Override embedding model (default: from config)
- `--api-key <key>`: API key for the embedding provider (if not provided, uses environment variables)
- `--clear-existing`: Clear existing collection before adding documents
- `--no-recursive`: Do not search for PDFs recursively in subdirectories
- `--no-embedding-suffix`: Disable automatic collection name suffix with embedding provider/model
- `--skip-if-exists`: Skip document ingestion if collection already exists

## Examples

### Basic vector store creation

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies
```

### Create vector store with custom chunk size

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies --chunk-size 1500 --chunk-overlap 300
```

### Clear existing collection and rebuild

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies --clear-existing
```

### Use specific embedding provider

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies --embedding-provider openai --embedding-model text-embedding-3-small
```

### Non-recursive folder search

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies --no-recursive
```

### Skip if collection already exists

```bash
sh scripts/jobs/create_hr_vectorstore.sh --folder /path/to/policies --skip-if-exists
```

## How It Works

1. The script automatically detects the project root directory
2. Changes to the project root directory
3. Executes `scripts/ingestion/create_vectorstore.py` with:
   - `--chatbot-type hr` (automatically added)
   - All other arguments you provide (passed through with `"$@"`)

## Configuration

The script uses configuration from `config/chatbot/hr_chatbot_config.yaml`. Most settings can be overridden via command-line arguments.

## Output

The script will:
- Load PDF documents from the specified folder
- Split them into chunks
- Create embeddings using the configured provider
- Store them in a ChromaDB vector store
- Persist the vector store to the configured directory

Logs will show:
- Configuration being used
- Number of documents loaded
- Number of chunks created
- Final vector store statistics

## Troubleshooting

### Script not found
Make sure you're running the script from the project root directory, or use an absolute path.

### Permission denied
Make the script executable:
```bash
chmod +x scripts/jobs/create_hr_vectorstore.sh
```

### No PDFs found
- Verify the folder path is correct
- Check that PDF files exist in the folder
- Use `--no-recursive` if you only want to search the top-level directory

### Collection already exists
- Use `--clear-existing` to replace the existing collection
- Use `--skip-if-exists` to skip ingestion if collection exists
- Or remove the existing collection manually

### API key errors
- Set environment variables for your embedding provider (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`)
- Or use `--api-key` to provide the key directly

## Related Files

- `scripts/ingestion/create_vectorstore.py`: The underlying Python script that performs the actual work
- `config/chatbot/hr_chatbot_config.yaml`: Configuration file for the HR chatbot
- `src/application/ingestion/`: Ingestion modules (loader, chunker, embedder)

