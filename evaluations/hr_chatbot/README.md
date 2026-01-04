# HR Chatbot Evaluation Guide

This guide explains how to evaluate the HR chatbot using LangSmith's LLM-as-judge evaluation framework.

## Overview

The evaluation script (`evaluate_hr_chatbot.py`) evaluates the HR chatbot on four key metrics:

1. **Correctness**: Measures how accurate the chatbot's answer is compared to a ground truth reference answer
2. **Groundedness**: Checks if the answer is based on the retrieved documents and doesn't contain hallucinations
3. **Relevance**: Evaluates whether the answer addresses the user's question
4. **Retrieval Relevance**: Assesses whether the retrieved documents are relevant to the question

## Prerequisites

1. **LangSmith Setup**: 
   - Set `LANGSMITH_API_KEY` in your `.env` file or environment
   - Set `LANGCHAIN_TRACING_V2=true` (or the script will force enable it)
   - Set `LANGCHAIN_PROJECT` (defaults to "rag-chatbot" from settings)

2. **OpenAI API Key**: 
   - Required for both the chatbot and the evaluator LLMs
   - Set `OPENAI_API_KEY` in your `.env` file

3. **HR Vector Store**:
   - Ensure the HR vector store is created and populated
   - Run `python jobs/create_vectorstore.py --chatbot-type hr` if needed

## Basic Usage

### Run with Default Examples

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py
```

This will:
- Create a default dataset with sample questions
- Run the evaluation
- Display results in the terminal (if pandas is installed)
- Show a link to view detailed results in LangSmith

### Run with Custom Dataset Name

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "HR Chatbot Q&A v2" --experiment-prefix "hr-chatbot-v2"
```

### Run with JSON File

```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-file sample_dataset.json --dataset-name "HR Chatbot from JSON"
```

**For more detailed information on all ways to pass datasets, see [DATASET_GUIDE.md](DATASET_GUIDE.md)**

## Creating Custom Evaluation Datasets

### Option 1: Programmatically Create Dataset

```python
from langsmith import Client
from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation

client = Client()

# Define your evaluation examples
examples = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days as per company policy."},
    },
    {
        "inputs": {"question": "What are the leave policies for full-time employees?"},
        "outputs": {"answer": "Full-time employees are entitled to 20 days of annual leave, 10 days of sick leave, and 5 days of casual leave per year."},
    },
    # Add more examples...
]

# Run evaluation
run_evaluation(
    dataset_name="HR Chatbot Custom Evaluation",
    examples=examples
)
```

### Option 2: Create Dataset in LangSmith UI

1. Go to [LangSmith](https://smith.langchain.com)
2. Navigate to "Datasets"
3. Create a new dataset
4. Add examples with:
   - `inputs`: `{"question": "your question here"}`
   - `outputs`: `{"answer": "expected answer here"}`

Then run:
```bash
python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "Your Dataset Name"
```

## Understanding the Results

### Metrics Explained

- **Correctness**: Percentage of answers that match the ground truth
  - `True`: Answer is factually correct relative to ground truth
  - `False`: Answer contains errors or conflicts with ground truth

- **Groundedness**: Percentage of answers that are based on retrieved documents
  - `True`: Answer is fully supported by the retrieved documents
  - `False`: Answer contains information not in the documents (hallucination)

- **Relevance**: Percentage of answers that address the question
  - `True`: Answer is helpful and relevant to the question
  - `False`: Answer doesn't address the question or is unhelpful

- **Retrieval Relevance**: Percentage of retrieved document sets that are relevant
  - `True`: Retrieved documents contain information related to the question
  - `False`: Retrieved documents are unrelated to the question

### Viewing Results

1. **In Terminal**: If pandas is installed, results are displayed as a summary table
2. **In LangSmith**: Click the experiment URL shown in the output to view:
   - Detailed scores for each example
   - Explanations from the LLM judges
   - Comparison across different runs
   - Performance trends over time

## Customizing Evaluators

You can modify the evaluator prompts in `evaluate_hr_chatbot.py` to adjust the evaluation criteria:

- `CORRECTNESS_INSTRUCTIONS`: Criteria for correctness evaluation
- `RELEVANCE_INSTRUCTIONS`: Criteria for relevance evaluation
- `GROUNDED_INSTRUCTIONS`: Criteria for groundedness evaluation
- `RETRIEVAL_RELEVANCE_INSTRUCTIONS`: Criteria for retrieval relevance evaluation

## Best Practices

1. **Create Representative Datasets**:
   - Include questions that cover different aspects of your HR policies
   - Include edge cases and ambiguous questions
   - Include questions that should return "I don't know" responses

2. **Use Clear Ground Truth Answers**:
   - Provide specific, accurate reference answers
   - Include context when necessary
   - Avoid ambiguous or overly broad answers

3. **Regular Evaluation**:
   - Run evaluations after major changes to the chatbot
   - Track performance over time using experiment prefixes
   - Compare different model configurations

4. **Iterate Based on Results**:
   - If groundedness is low: Check retrieval quality or add more context
   - If relevance is low: Review system prompts and instructions
   - If retrieval relevance is low: Improve chunking or embedding strategy

## Troubleshooting

### "Dataset not found" Error
- Ensure the dataset name matches exactly (case-sensitive)
- Create the dataset first in LangSmith UI or via the script

### "Vector store not found" Error
- Run `python jobs/create_vectorstore.py --chatbot-type hr` to create the vector store
- Check that the vector store configuration in `hr_chatbot_config.yaml` is correct

### "API Key not found" Error
- Ensure `LANGSMITH_API_KEY` and `OPENAI_API_KEY` are set in `.env`
- Check that the API keys are valid

### Low Scores
- Review individual examples in LangSmith to understand failures
- Check if ground truth answers are accurate
- Verify that the vector store contains relevant documents
- Consider adjusting evaluator prompts if criteria are too strict

## Reference

- [LangSmith RAG Evaluation Tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
- [LangSmith Evaluation Documentation](https://docs.langchain.com/langsmith/evaluate)

