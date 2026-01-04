# How to Pass Datasets to HR Chatbot Evaluation

This guide explains all the different ways you can pass datasets to the evaluation script.

## Method 1: Command Line - Use Existing LangSmith Dataset

If you've already created a dataset in LangSmith (via UI or programmatically), simply pass the dataset name:

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "My HR Dataset"
```

The script will use the existing dataset from LangSmith.

## Method 2: Command Line - Create Dataset with Default Examples

If you don't specify a dataset name, the script will create one with default examples:

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py
```

This creates a dataset named "HR Chatbot Q&A" with default examples.

## Method 3: Programmatically - Pass Examples Directly

You can pass examples directly when calling `run_evaluation()`:

```python
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

# Define your examples
examples = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days."},
    },
    {
        "inputs": {"question": "What are the leave policies for full-time employees?"},
        "outputs": {"answer": "Full-time employees are entitled to 20 days of annual leave, 10 days of sick leave, and 5 days of casual leave per year."},
    },
]

# Run evaluation - this will create a new dataset with your examples
results, df = run_evaluation(
    dataset_name="HR Chatbot Custom Evaluation",
    examples=examples
)
```

## Method 4: Use Sample Datasets

You can use the predefined sample datasets:

```python
from evaluations.hr_chatbot.sample_evaluation_dataset import get_dataset_by_name
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

# Get a predefined dataset
examples = get_dataset_by_name("detailed")  # Options: "basic", "detailed", "edge_case", "complex"

# Run evaluation
results, df = run_evaluation(
    dataset_name="HR Chatbot Detailed Evaluation",
    examples=examples
)
```

## Method 5: Load Dataset from JSON File (Command Line)

You can pass a JSON file directly via command line:

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-file sample_dataset.json --dataset-name "HR Chatbot from JSON"
```

The JSON file should contain an array of examples in this format:
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
      "answer": "Leave policies include annual leave, sick leave, and other types."
    }
  }
]
```

See `sample_dataset.json` for a complete example.

## Method 6: Load Dataset from File (Programmatically)

You can load examples from a JSON or Python file programmatically:

```python
import json
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

# Load from JSON file
with open("my_dataset.json", "r") as f:
    examples = json.load(f)

# Or load from Python file
from my_custom_dataset import MY_HR_DATASET
examples = MY_HR_DATASET

# Run evaluation
results, df = run_evaluation(
    dataset_name="HR Chatbot from File",
    examples=examples
)
```

## Method 7: Create Dataset First, Then Evaluate

You can create the dataset separately and then use it:

```python
from langsmith import Client
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

client = Client()

# Create dataset
examples = [
    {
        "inputs": {"question": "What is the notice period?"},
        "outputs": {"answer": "The notice period is 30 days."},
    },
]

dataset = client.create_dataset(dataset_name="My Pre-created Dataset")
client.create_examples(dataset_id=dataset.id, examples=examples)

# Later, use it for evaluation
results, df = run_evaluation(
    dataset_name="My Pre-created Dataset"
)
```

## Dataset Format

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

**Important Notes:**
- `inputs` must contain a `"question"` key
- `outputs` must contain an `"answer"` key (used for correctness evaluation)
- Each example represents one question-answer pair

## Complete Example

Here's a complete example showing how to create and use a custom dataset:

```python
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

# Define your HR policy evaluation dataset
hr_evaluation_examples = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days as per company policy."},
    },
    {
        "inputs": {"question": "What are the leave policies for full-time employees?"},
        "outputs": {"answer": "Full-time employees are entitled to 20 days of annual leave, 10 days of sick leave, and 5 days of casual leave per year."},
    },
    {
        "inputs": {"question": "How do I apply for a promotion?"},
        "outputs": {"answer": "Employees can request a promotion by discussing with their manager, submitting a formal application through the HR portal, and going through the performance review process."},
    },
]

# Run the evaluation
results, df = run_evaluation(
    dataset_name="HR Chatbot Production Evaluation v1",
    experiment_prefix="hr-chatbot-prod-v1",
    examples=hr_evaluation_examples
)

# View results
print(f"View results: {results.experiment_url}")
if df is not None:
    print(df)
```

## Tips

1. **Dataset Names**: Use descriptive names that include version numbers (e.g., "HR Chatbot Q&A v2") to track different evaluation runs
2. **Reuse Datasets**: Once created in LangSmith, you can reuse the same dataset name for multiple evaluations
3. **Update Datasets**: If you need to update a dataset, delete it in LangSmith UI first, or use a new name
4. **Large Datasets**: For large datasets, consider creating them in LangSmith UI or programmatically first, then reference by name

