# Evaluations Module

This module contains evaluation scripts and utilities for evaluating chatbots using LLM-as-Judge evaluation with LangSmith.

## Structure

```
evaluations/
├── core/                    # Generic evaluation framework (reusable for any chatbot)
│   ├── __init__.py
│   └── evaluator.py         # ChatbotEvaluator class and evaluation functions
├── hr_chatbot/              # HR chatbot-specific evaluation
│   ├── __init__.py
│   ├── evaluate_hr_chatbot.py    # HR evaluation script (uses generic framework)
│   ├── evaluate_hr_chatbot.sh    # Shell script for running HR evaluation
│   ├── hr_evaluation_dataset.py  # HR-specific evaluation datasets
│   ├── sample_dataset.json       # Sample evaluation dataset
│   └── README.md                 # HR evaluation documentation
└── README.md                # This file
```

## Generic Evaluation Framework

The `evaluations/core/` module provides a reusable evaluation framework that can evaluate any chatbot following the `ChatbotAgent` interface.

### Key Components

- **`ChatbotEvaluator`**: Main class for evaluating chatbots
- **Evaluators**: Four built-in evaluators:
  - **Correctness**: Compares response against ground truth reference
  - **Groundedness**: Checks if response is based on retrieved documents
  - **Relevance**: Checks if response addresses the question
  - **Retrieval Relevance**: Checks if retrieved documents are relevant

### Usage Example

```python
from evaluations.core import ChatbotEvaluator
from src.domain.chatbot.hr_chatbot import get_hr_chatbot

# Create evaluator
evaluator = ChatbotEvaluator(
    chatbot_getter=get_hr_chatbot,
    chatbot_type="hr",
    config_filename="hr_chatbot_config.yaml"
)

# Run evaluation
results, df = evaluator.run_evaluation(
    dataset_name="HR Chatbot Q&A",
    examples=[
        {
            "inputs": {"question": "What is the notice period?"},
            "outputs": {"answer": "The notice period is 30 days."}
        }
    ]
)
```

## HR Chatbot Evaluation

The `evaluations/hr_chatbot/` module provides HR-specific evaluation scripts and datasets.

### Running HR Evaluation

#### Using Python Script

```bash
# With default dataset
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
# Make script executable (first time only)
chmod +x evaluations/hr_chatbot/evaluate_hr_chatbot.sh

# Run evaluation
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh

# With options
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh --dataset-name "HR Chatbot Q&A"
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh --dataset-file evaluations/hr_chatbot/sample_dataset.json
./evaluations/hr_chatbot/evaluate_hr_chatbot.sh --experiment-prefix "hr-eval-v2"
```

### Programmatic Usage

```python
from evaluations.hr_chatbot import run_evaluation

# Run evaluation with custom examples
custom_examples = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days."}
    }
]

results, df = run_evaluation(
    dataset_name="HR Chatbot Custom Evaluation",
    examples=custom_examples
)
```

## Creating a New Chatbot Evaluation

To create evaluation for a new chatbot:

1. Create a new folder in `evaluations/` (e.g., `evaluations/support_chatbot/`)

2. Create an evaluation script that uses the generic framework:

```python
# evaluations/support_chatbot/evaluate_support_chatbot.py
from evaluations.core import ChatbotEvaluator
from src.domain.chatbot.support_chatbot import get_support_chatbot

def _get_support_evaluator():
    return ChatbotEvaluator(
        chatbot_getter=get_support_chatbot,
        chatbot_type="support",
        config_filename="support_chatbot_config.yaml"
    )

def run_evaluation(dataset_name=None, examples=None):
    evaluator = _get_support_evaluator()
    return evaluator.run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix="support-chatbot-rag-eval",
        examples=examples
    )

if __name__ == "__main__":
    # Add CLI argument parsing...
    pass
```

3. Optionally create a shell script similar to `evaluate_hr_chatbot.sh`

## Evaluation Metrics

The framework evaluates chatbots on four metrics:

1. **Correctness**: How accurate is the answer compared to the ground truth?
2. **Groundedness**: Is the answer based on the retrieved documents (no hallucinations)?
3. **Relevance**: Does the answer address the question?
4. **Retrieval Relevance**: Are the retrieved documents relevant to the question?

All metrics are evaluated using LLM-as-Judge with structured output for consistency.

## Requirements

- LangSmith account and API key configured
- LangChain and LangSmith packages installed
- Chatbot implementation following `ChatbotAgent` interface
- Vector store configured for the chatbot type

## See Also

- [HR Chatbot Evaluation README](hr_chatbot/README.md) - Detailed HR evaluation documentation
- [LangSmith Evaluation Tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial) - Original tutorial this is based on

