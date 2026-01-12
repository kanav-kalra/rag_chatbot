# Evaluation Test Cases Documentation

This directory contains comprehensive test cases for all evaluation metrics used in the chatbot evaluation framework.

## Overview

The evaluation framework includes 5 metrics:

1. **Correctness**: Factual accuracy against ground truth
2. **Relevance**: Answer helpfulness and relevance to question
3. **Groundedness**: Answer based on retrieved documents with proper citations
4. **Retrieval Relevance**: Retrieved documents relevant to question
5. **Scannability**: Visual structure with headers and bullet points

## Files

### `evaluation_test_cases.py`
Contains comprehensive test cases for all metrics, organized by metric type. Each test case includes:
- Test name and description
- Input question
- Expected output (for correctness)
- Sample answer (student answer)
- Retrieved documents (for groundedness and retrieval relevance)
- Expected metric scores (True/False)

### `test_evaluator_metrics.py`
Test runner script that validates evaluators work correctly by running test cases against the evaluator.

### `evaluator.py`
Main evaluator class with all metric implementations. Updated to include:
- `ScannabilityGrade` TypedDict schema
- `scannability()` evaluator method
- Scannability included in default evaluators list

## Test Cases Summary

### Correctness (6 test cases)
- ✅ Positive: Exact match with ground truth
- ✅ Positive: Semantic match with minor wording differences
- ✅ Positive: Correctly admits information gap
- ❌ Negative: Wrong numbers/dates
- ❌ Negative: Missing important information
- ❌ Negative: Hallucinates when info is missing

### Relevance (5 test cases)
- ✅ Positive: Direct answer with actionable steps
- ✅ Positive: Complete information (eligibility, duration, process)
- ❌ Negative: Off-topic answers
- ❌ Negative: Incomplete information
- ❌ Negative: Vague/unhelpful answers

### Groundedness (7 test cases)
- ✅ Positive: Proper citations with single source
- ✅ Positive: Multiple sources with proper citations
- ❌ Negative: Hallucinated information
- ❌ Negative: Missing citation markers
- ❌ Negative: Filename in body text
- ❌ Negative: Duplicate sources
- ❌ Negative: Unsupported claims

### Retrieval Relevance (6 test cases)
- ✅ Positive: Exact keyword match
- ✅ Positive: Semantic match (synonyms)
- ✅ Positive: Category match (HR categories)
- ✅ Positive: Relevant with some noise (low bar)
- ❌ Negative: Completely unrelated documents
- ❌ Negative: Wrong category

### Scannability (6 test cases)
- ✅ Positive: Headers and bullet points
- ✅ Positive: Step-by-step format
- ❌ Negative: Dense paragraphs
- ❌ Negative: No headers
- ❌ Negative: No bullet points
- ❌ Negative: Mixed format (header but dense text)

**Total: 30 test cases**

## Usage

### Running Test Cases

```bash
# List all available test cases
python evaluations/core/test_evaluator_metrics.py --list

# Test all metrics
python evaluations/core/test_evaluator_metrics.py

# Test a specific metric
python evaluations/core/test_evaluator_metrics.py --metric correctness

# Verbose output
python evaluations/core/test_evaluator_metrics.py --metric relevance --verbose
```

### Using Test Cases in Code

```python
from evaluations.core.evaluation_test_cases import (
    get_test_cases_for_metric,
    get_all_test_cases,
    get_test_case_summary
)

# Get test cases for a specific metric
correctness_cases = get_test_cases_for_metric("correctness")

# Get all test cases
all_cases = get_all_test_cases()

# Get summary
summary = get_test_case_summary()
```

### Using Test Cases with Evaluator

```python
from evaluations.core.evaluator import ChatbotEvaluator
from evaluations.core.evaluation_test_cases import get_test_cases_for_metric

# Create evaluator
evaluator = ChatbotEvaluator(
    chatbot_getter=your_chatbot_getter,
    chatbot_type="hr",
    config_filename="hr_chatbot_config.yaml"
)

# Get test cases
test_cases = get_test_cases_for_metric("correctness")

# Run a test case
for test_case in test_cases:
    score = evaluator.correctness(
        test_case["inputs"],
        test_case["outputs"],
        test_case["reference_outputs"]
    )
    print(f"Test: {test_case['name']}, Score: {score}")
```

## Metric Details

### Correctness
Evaluates factual accuracy by comparing against ground truth. Checks:
- Numbers, dates, and names match
- No contradictory statements
- Proper handling of information gaps

### Relevance
Evaluates if answer addresses the question. Checks:
- Direct relevance to user intent
- Completeness (covers all aspects)
- Actionable information

### Groundedness
Evaluates if answer is based on retrieved documents. Checks:
- All claims supported by facts
- Proper citation format ([1], [2], etc.)
- No filenames in body text
- Sources section with unique filenames
- No unsupported claims

### Retrieval Relevance
Evaluates if retrieved documents are relevant. Checks:
- Semantic alignment with question
- Topic consistency (HR categories)
- Keyword matching
- Low bar (some noise OK if core info present)

### Scannability
Evaluates visual structure and readability. Checks:
- Bold headers to separate sections
- Bullet points for details/steps
- No dense paragraphs for multiple facts

## Adding New Test Cases

To add new test cases, edit `evaluation_test_cases.py` and add to the appropriate list:

```python
NEW_TEST_CASE = {
    "name": "descriptive_name",
    "inputs": {"question": "..."},
    "outputs": {"answer": "..."},
    "expected_score": True,  # or False
    "description": "What this test case validates"
}
```

For correctness, also include:
```python
"reference_outputs": {"answer": "..."}
```

For groundedness and retrieval_relevance, also include:
```python
"documents": [Document(page_content="...", metadata={...})]
```

## Notes

- Test cases are designed to validate both positive (should pass) and negative (should fail) scenarios
- The evaluator uses LLM-as-Judge, so results may vary slightly
- Test cases should be updated as evaluation criteria evolve
- All metrics now include scannability in the default evaluators list

