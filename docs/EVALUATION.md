# Evaluation Guide

This guide covers how to evaluate chatbots using the LLM-as-Judge evaluation framework with LangSmith.

## Overview

The evaluation system uses **LLM-as-Judge** evaluation to assess chatbot performance on four key metrics:

1. **Correctness**: How accurate is the answer compared to the ground truth?
2. **Groundedness**: Is the answer based on retrieved documents (no hallucinations)?
3. **Relevance**: Does the answer address the question?
4. **Retrieval Relevance**: Are the retrieved documents relevant to the question?

### Architecture

```
evaluations/
├── core/                    # Generic evaluation framework
│   └── evaluator.py        # ChatbotEvaluator class (reusable)
├── hr_chatbot/              # HR chatbot-specific evaluation
│   ├── evaluate_hr_chatbot.py
│   ├── evaluate_hr_chatbot.sh
│   ├── hr_evaluation_dataset.py
│   └── sample_dataset.json
└── README.md                # Evaluation overview
```

The framework is **generic** and can evaluate any chatbot that follows the `ChatbotAgent` interface.

## Prerequisites

### 1. LangSmith Setup

1. **Create LangSmith Account**: Sign up at [LangSmith](https://smith.langchain.com)
2. **Get API Key**: Get your API key from LangSmith settings
3. **Configure Environment**:
   ```bash
   # In .env file
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=rag-chatbot  # Optional, defaults to this
   ```

### 2. API Keys

- **OpenAI API Key**: Required for evaluator LLMs
- **Chatbot API Keys**: Required for the chatbot being evaluated

```bash
OPENAI_API_KEY=your-openai-api-key
# Plus any other keys needed for your chatbot (GOOGLE_API_KEY, etc.)
```

### 3. Vector Store

Ensure the vector store for your chatbot is created:

```bash
# For HR chatbot
./scripts/jobs/create_hr_vectorstore.sh --folder-path /path/to/documents
```

## Quick Start

### Running HR Chatbot Evaluation

#### Using Python Script

```bash
# With default examples
python evaluations/hr_chatbot/evaluate_hr_chatbot.py

# With existing dataset
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "HR Chatbot Q&A"

# With JSON file
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-file evaluations/hr_chatbot/sample_dataset.json

# With custom experiment prefix
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --experiment-prefix "hr-eval-v2"
```

#### Using Shell Script

```bash
# Make executable (first time only)
chmod +x evaluations/hr_chatbot/evaluate_hr_chatbot.sh

# Run evaluation
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh

# With options
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh --dataset-name "HR Chatbot Q&A"
```

### Programmatic Usage

```python
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

# Define evaluation examples
examples = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days."}
    },
    {
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {"answer": "Full-time employees are entitled to 20 days of annual leave..."}
    }
]

# Run evaluation
results, df = run_evaluation(
    dataset_name="HR Chatbot Custom Evaluation",
    experiment_prefix="hr-chatbot-custom",
    examples=examples
)

# View results
print(f"View results: {results.experiment_url}")
if df is not None:
    print(df)
```

## Understanding Results

### Metrics Explained

#### 1. Correctness

**What it measures**: How accurate the answer is compared to the ground truth reference.

- **True**: Answer is factually correct relative to ground truth
- **False**: Answer contains errors or conflicts with ground truth

**Example**:
- **Question**: "What is the notice period?"
- **Ground Truth**: "The notice period is 30 days."
- **Chatbot Answer**: "The notice period is 30 days as per company policy."
- **Result**: ✅ **True** (factually correct, even with extra information)

#### 2. Groundedness

**What it measures**: Whether the answer is based on retrieved documents (no hallucinations).

- **True**: Answer is fully supported by retrieved documents
- **False**: Answer contains information not in the documents (hallucination)

**Example**:
- **Question**: "What is the vacation policy?"
- **Retrieved Docs**: "Employees accrue 1.25 days per month."
- **Chatbot Answer**: "Employees accrue 1.25 days per month."
- **Result**: ✅ **True** (fully grounded in documents)

#### 3. Relevance

**What it measures**: Whether the answer addresses the user's question.

- **True**: Answer is helpful and relevant to the question
- **False**: Answer doesn't address the question or is unhelpful

**Example**:
- **Question**: "What is the vacation policy?"
- **Chatbot Answer**: "The vacation policy allows employees to accrue 1.25 days per month..."
- **Result**: ✅ **True** (directly addresses the question)

#### 4. Retrieval Relevance

**What it measures**: Whether the retrieved documents are relevant to the question.

- **True**: Retrieved documents contain information related to the question
- **False**: Retrieved documents are unrelated to the question

**Example**:
- **Question**: "What is the vacation policy?"
- **Retrieved Docs**: Documents about vacation accrual, leave types, etc.
- **Result**: ✅ **True** (documents are relevant)

### Viewing Results

#### In Terminal

If pandas is installed, results are displayed as a summary table:

```
Metric              Score
Correctness         0.85
Groundedness        0.90
Relevance           0.95
Retrieval Relevance 0.88
```

#### In LangSmith

Click the experiment URL shown in the output to view:
- **Detailed scores** for each example
- **LLM judge explanations** for each evaluation
- **Comparison** across different runs
- **Performance trends** over time
- **Failed examples** with detailed analysis

## Creating Evaluation Datasets

### Method 1: Programmatically

```python
from evaluations.hr_chatbot.evaluate_hr_chatbot import create_evaluation_dataset

examples = [
    {
        "inputs": {"question": "What is the notice period?"},
        "outputs": {"answer": "The notice period is 30 days."}
    },
    # ... more examples
]

dataset_name = create_evaluation_dataset(
    dataset_name="HR Chatbot Q&A v2",
    examples=examples,
    overwrite=False  # Set to True to overwrite existing dataset
)
```

### Method 2: JSON File

Create a JSON file with examples:

```json
[
  {
    "inputs": {
      "question": "What is the notice period for grade 8 employees?"
    },
    "outputs": {
      "answer": "The notice period for grade 8 employees is 30 days."
    }
  },
  {
    "inputs": {
      "question": "What are the leave policies?"
    },
    "outputs": {
      "answer": "Full-time employees are entitled to 20 days of annual leave..."
    }
  }
]
```

Then use it:
```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py \
  --dataset-file my_dataset.json \
  --dataset-name "My Custom Dataset"
```

### Method 3: LangSmith UI

1. Go to [LangSmith](https://smith.langchain.com)
2. Navigate to "Datasets"
3. Click "Create Dataset"
4. Add examples with:
   - `inputs`: `{"question": "your question"}`
   - `outputs`: `{"answer": "expected answer"}`

Then use it:
```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "Your Dataset Name"
```

### Dataset Format

All datasets must follow this format:

```python
examples = [
    {
        "inputs": {
            "question": "Your question here"
        },
        "outputs": {
            "answer": "Expected/ground truth answer here"
        }
    },
    # ... more examples
]
```

**Important**:
- `inputs` must contain a `"question"` key
- `outputs` must contain an `"answer"` key (used for correctness evaluation)
- Each example represents one question-answer pair

## Creating Evaluations for New Chatbots

To create evaluation for a new chatbot:

### Step 1: Create Evaluation Script

Create `evaluations/{chatbot_type}/evaluate_{chatbot_type}_chatbot.py`:

```python
"""
{ChatbotType} Chatbot Evaluation
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluations.core.evaluator import ChatbotEvaluator
from src.domain.chatbot.{chatbot_type}_chatbot import get_{chatbot_type}_chatbot

# Create evaluator instance
_evaluator = None

def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = ChatbotEvaluator(
            chatbot_getter=get_{chatbot_type}_chatbot,
            chatbot_type="{chatbot_type}",
            config_filename="{chatbot_type}_chatbot_config.yaml",
            retrieval_k=6
        )
    return _evaluator

def run_evaluation(dataset_name=None, experiment_prefix="{chatbot_type}-chatbot-rag-eval", examples=None):
    evaluator = _get_evaluator()
    return evaluator.run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        examples=examples
    )

if __name__ == "__main__":
    # Add CLI argument parsing similar to evaluate_hr_chatbot.py
    pass
```

### Step 2: Create Sample Dataset (Optional)

Create `evaluations/{chatbot_type}/sample_dataset.json` with sample examples.

### Step 3: Create Shell Script (Optional)

Create `evaluations/{chatbot_type}/evaluate_{chatbot_type}_chatbot.sh`:

```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$PROJECT_ROOT"
python evaluations/{chatbot_type}/evaluate_{chatbot_type}_chatbot.py "$@"
```

Make it executable:
```bash
chmod +x evaluations/{chatbot_type}/evaluate_{chatbot_type}_chatbot.sh
```

### Step 4: Run Evaluation

```bash
python evaluations/{chatbot_type}/evaluate_{chatbot_type}_chatbot.py
```

## Customizing Evaluators

You can customize evaluator prompts to adjust evaluation criteria. Modify the prompts in `evaluations/core/evaluator.py`:

- **`CORRECTNESS_INSTRUCTIONS`**: Criteria for correctness evaluation
- **`RELEVANCE_INSTRUCTIONS`**: Criteria for relevance evaluation
- **`GROUNDED_INSTRUCTIONS`**: Criteria for groundedness evaluation
- **`RETRIEVAL_RELEVANCE_INSTRUCTIONS`**: Criteria for retrieval relevance evaluation

Or override them when creating the evaluator:

```python
from evaluations.core.evaluator import ChatbotEvaluator

evaluator = ChatbotEvaluator(
    chatbot_getter=get_hr_chatbot,
    chatbot_type="hr",
    config_filename="hr_chatbot_config.yaml",
    correctness_instructions="Your custom correctness criteria...",
    relevance_instructions="Your custom relevance criteria...",
    # ... other custom instructions
)
```

## Best Practices

### 1. Create Representative Datasets

- Include questions covering different aspects of your domain
- Include edge cases and ambiguous questions
- Include questions that should return "I don't know" responses
- Balance easy and difficult questions

### 2. Use Clear Ground Truth Answers

- Provide specific, accurate reference answers
- Include context when necessary
- Avoid ambiguous or overly broad answers
- Match the expected format and detail level

### 3. Regular Evaluation

- Run evaluations after major changes to the chatbot
- Track performance over time using experiment prefixes
- Compare different model configurations
- Monitor for regressions

### 4. Iterate Based on Results

**If Correctness is Low**:
- Review ground truth answers for accuracy
- Check if chatbot is understanding questions correctly
- Verify model configuration and prompts

**If Groundedness is Low**:
- Check retrieval quality
- Review system prompts to emphasize grounding
- Add more context to retrieved documents
- Verify vector store contains relevant documents

**If Relevance is Low**:
- Review system prompts and instructions
- Check if chatbot is understanding questions
- Verify agent instructions are clear

**If Retrieval Relevance is Low**:
- Improve chunking strategy (size, overlap)
- Review embedding model choice
- Check document quality and organization
- Consider adding metadata for better filtering

### 5. Track Experiments

Use descriptive experiment prefixes to track different runs:

```bash
# Version tracking
--experiment-prefix "hr-chatbot-v1.0"
--experiment-prefix "hr-chatbot-v1.1"

# Configuration tracking
--experiment-prefix "hr-chatbot-gpt4"
--experiment-prefix "hr-chatbot-gemini"

# Feature tracking
--experiment-prefix "hr-chatbot-with-summarization"
--experiment-prefix "hr-chatbot-baseline"
```

## Troubleshooting

### "Dataset not found" Error

**Problem**: Dataset doesn't exist in LangSmith.

**Solutions**:
- Ensure dataset name matches exactly (case-sensitive)
- Create the dataset first in LangSmith UI or via script
- Check that `LANGSMITH_API_KEY` is set correctly

### "Vector store not found" Error

**Problem**: Vector store doesn't exist for the chatbot.

**Solutions**:
- Create vector store: `python scripts/ingestion/create_vectorstore.py --chatbot-type {type}`
- Check `persist_dir` in chatbot config matches actual location
- Verify collection name is correct

### "API Key not found" Error

**Problem**: Missing or invalid API keys.

**Solutions**:
- Ensure `LANGSMITH_API_KEY` is set in `.env`
- Ensure `OPENAI_API_KEY` is set (for evaluator LLMs)
- Check that API keys are valid and not expired
- Verify keys are loaded (check logs)

### Low Scores

**Problem**: Evaluation scores are consistently low.

**Solutions**:
- Review individual examples in LangSmith to understand failures
- Check if ground truth answers are accurate
- Verify vector store contains relevant documents
- Consider adjusting evaluator prompts if criteria are too strict
- Review chatbot configuration (model, prompts, etc.)

### Evaluation Takes Too Long

**Problem**: Evaluation runs are slow.

**Solutions**:
- Reduce dataset size for testing
- Use faster LLM models for evaluators
- Run evaluations in parallel (if supported)
- Check network connectivity to LangSmith

## Advanced Usage

### Custom Evaluation Metrics

You can add custom evaluators by extending `ChatbotEvaluator`:

```python
from evaluations.core.evaluator import ChatbotEvaluator
from langsmith import evaluate

class CustomEvaluator(ChatbotEvaluator):
    def custom_metric(self, run, example):
        # Your custom evaluation logic
        return {"score": 0.95, "reasoning": "..."}

# Use custom evaluator
evaluator = CustomEvaluator(...)
results = evaluator.run_evaluation(...)
```

### Batch Evaluation

Evaluate multiple chatbot configurations:

```python
configs = [
    {"model": "gpt-4", "temperature": 0.7},
    {"model": "gemini-2.5-flash", "temperature": 0.2},
]

for config in configs:
    # Update chatbot config
    # Run evaluation
    results = run_evaluation(
        experiment_prefix=f"hr-chatbot-{config['model']}"
    )
```

## Related Documentation

- [HR Chatbot Evaluation README](../evaluations/hr_chatbot/README.md) - Detailed HR evaluation guide
- [Dataset Guide](../evaluations/hr_chatbot/DATASET_GUIDE.md) - All ways to pass datasets
- [Evaluations README](../evaluations/README.md) - Evaluation framework overview
- [LangSmith Documentation](https://docs.langchain.com/langsmith/evaluate) - Official LangSmith docs

## Reference

- [LangSmith RAG Evaluation Tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
- [LangSmith Evaluation Documentation](https://docs.langchain.com/langsmith/evaluate)

