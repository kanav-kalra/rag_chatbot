#!/bin/bash
# Shell script wrapper for creating HR chatbot vector store
# This script makes it easy to run the vector store creation job

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up two levels: scripts/jobs -> scripts -> project_root
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Run the Python script with chatbot-type=hr and all passed arguments
python scripts/ingestion/create_vectorstore.py --chatbot-type hr "$@"

