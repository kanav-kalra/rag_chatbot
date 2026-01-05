#!/bin/bash
#
# HR Chatbot Evaluation Script
#
# This script runs evaluation on the HR chatbot using LangSmith.
# It supports various options for dataset creation and evaluation.
#
# Usage:
#   ./evaluate_hr_chatbot.sh [OPTIONS]
#
# Options:
#   --dataset-name NAME       Use or create a dataset with the given name
#   --dataset-file PATH       Load examples from a JSON file
#   --experiment-prefix NAME  Set the experiment prefix (default: hr-chatbot-rag-eval)
#   --help                    Show this help message
#
# Examples:
#   # Run with default dataset
#   ./evaluate_hr_chatbot.sh
#
#   # Run with existing dataset
#   ./evaluate_hr_chatbot.sh --dataset-name "HR Chatbot Q&A"
#
#   # Run with examples from JSON file
#   ./evaluate_hr_chatbot.sh --dataset-file evaluations/hr_chatbot/sample_dataset.json
#
#   # Run with custom experiment prefix
#   ./evaluate_hr_chatbot.sh --experiment-prefix "hr-eval-v2"
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default values
DATASET_NAME=""
DATASET_FILE=""
EXPERIMENT_PREFIX="hr-chatbot-rag-eval"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-file)
            DATASET_FILE="$2"
            shift 2
            ;;
        --experiment-prefix)
            EXPERIMENT_PREFIX="$2"
            shift 2
            ;;
        --help)
            echo "HR Chatbot Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset-name NAME       Use or create a dataset with the given name"
            echo "  --dataset-file PATH       Load examples from a JSON file"
            echo "  --experiment-prefix NAME  Set the experiment prefix (default: hr-chatbot-rag-eval)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --dataset-name \"HR Chatbot Q&A\""
            echo "  $0 --dataset-file evaluations/hr_chatbot/sample_dataset.json"
            echo "  $0 --experiment-prefix \"hr-eval-v2\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build Python command
PYTHON_CMD="python evaluations/hr_chatbot/evaluate_hr_chatbot.py"

if [ -n "$DATASET_NAME" ]; then
    PYTHON_CMD="$PYTHON_CMD --dataset-name \"$DATASET_NAME\""
fi

if [ -n "$DATASET_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --dataset-file \"$DATASET_FILE\""
fi

if [ -n "$EXPERIMENT_PREFIX" ]; then
    PYTHON_CMD="$PYTHON_CMD --experiment-prefix \"$EXPERIMENT_PREFIX\""
fi

# Run the evaluation
echo "Running HR Chatbot Evaluation..."
echo "Command: $PYTHON_CMD"
echo ""

eval $PYTHON_CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi

