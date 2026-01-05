"""
LLM-as-Judge Evaluation for HR Chatbot using LangSmith

This script evaluates the HR chatbot on four metrics:
1. Correctness: Response vs reference answer
2. Groundedness: Response vs retrieved docs
3. Relevance: Response vs input question
4. Retrieval relevance: Retrieved docs vs input question

Based on: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from uuid import uuid4

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langsmith import Client, traceable
from langchain_core.documents import Document

# Handle TypedDict and Annotated imports
# Python 3.9+ has Annotated in typing, but we use typing_extensions for compatibility
try:
    from typing_extensions import TypedDict, Annotated
except ImportError:
    # Fallback for Python 3.9+
    from typing import TypedDict, Annotated

from src.shared.config.settings import settings
from src.shared.config.logging import logger
from src.shared.config.langsmith import initialize_langsmith
from src.domain.chatbot.hr_chatbot import get_hr_chatbot
from src.domain.retrieval.service import RetrievalService
from src.infrastructure.vectorstore.manager import get_vector_store
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys
from src.infrastructure.llm.manager import get_llm_manager


# Initialize LangSmith
initialize_langsmith(force_enable=True)
client = Client()

# Load HR chatbot config to get model settings
_hr_config_manager = ChatbotConfigManager("hr_chatbot.yaml")


@traceable()
def hr_chatbot_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function for HR chatbot that returns both answer and retrieved documents.
    This is needed for evaluation as we need access to both the response and the documents.
    
    Args:
        inputs: Dictionary with "question" key
        
    Returns:
        Dictionary with "answer" and "documents" keys
    """
    question = inputs["question"]
    
    # Get HR chatbot instance
    chatbot = get_hr_chatbot()
    
    # Generate a unique thread_id for this evaluation run
    thread_id = f"eval_{uuid4().hex[:8]}"
    
    # Get the answer from the chatbot
    answer = chatbot.chat(
        query=question,
        thread_id=thread_id,
        user_id="evaluation_user"
    )
    
    # Get retrieved documents by directly calling the retrieval service
    # This ensures we have access to the documents for evaluation
    vector_store = get_vector_store("hr")
    retrieval_service = RetrievalService(vector_store)
    
    # Retrieve documents (using same k as the tool would)
    _, artifact = retrieval_service.retrieve(query=question, k=6)
    
    # Convert artifact to Document objects for consistency
    documents = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in artifact
    ]
    
    return {
        "answer": answer,
        "documents": documents
    }


# Grade output schemas for each evaluator
class CorrectnessGrade(TypedDict):
    """Schema for correctness evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    """Schema for relevance evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


class GroundedGrade(TypedDict):
    """Schema for groundedness evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


class RetrievalRelevanceGrade(TypedDict):
    """Schema for retrieval relevance evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grader LLMs - Use model from HR chatbot config
# Get model configuration from hr_chatbot.yaml
grader_model_name = _hr_config_manager.get(ConfigKeys.MODEL_NAME) or settings.CHAT_MODEL
grader_temperature = _hr_config_manager.get(ConfigKeys.MODEL_TEMPERATURE)
if grader_temperature is not None:
    grader_temperature = float(grader_temperature)
else:
    grader_temperature = 0  # Use 0 for evaluation (more deterministic)

logger.info(f"Using model from HR chatbot config for graders: {grader_model_name} (temperature: {grader_temperature})")

# Use LLM manager to create the grader LLM (respects model configuration)
# This ensures we use the same model provider as configured in hr_chatbot.yaml
llm_manager = get_llm_manager()
grader_llm = llm_manager.get_llm(
    model_name=grader_model_name,
    temperature=grader_temperature,
    max_tokens=None,  # Let model use default for evaluation
    use_cache=False  # Don't cache evaluation LLMs
)

# Note: Structured output might not work the same way with all models
# If it fails, we may need to adjust the evaluation approach
try:
    correctness_llm = grader_llm.with_structured_output(
        CorrectnessGrade, method="json_schema", strict=True
    )
    relevance_llm = grader_llm.with_structured_output(
        RelevanceGrade, method="json_schema", strict=True
    )
    grounded_llm = grader_llm.with_structured_output(
        GroundedGrade, method="json_schema", strict=True
    )
    retrieval_relevance_llm = grader_llm.with_structured_output(
        RetrievalRelevanceGrade, method="json_schema", strict=True
    )
    logger.info("Successfully created structured output graders")
except Exception as e:
    logger.warning(f"Failed to create structured output graders: {e}")
    logger.info("Falling back to regular LLM calls (may need manual parsing)")
    # Fallback: use regular LLM and parse responses manually if needed
    correctness_llm = relevance_llm = grounded_llm = retrieval_relevance_llm = grader_llm


# Evaluator prompts
CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


# Evaluator functions
def correctness(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> bool:
    """
    Evaluator for RAG answer accuracy.
    Compares the generated answer against a ground truth reference answer.
    
    Args:
        inputs: Input dictionary with "question" key
        outputs: Output dictionary with "answer" key
        reference_outputs: Reference dictionary with "answer" key
        
    Returns:
        True if the answer is correct, False otherwise
    """
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    
    # Run evaluator
    grade = correctness_llm.invoke([
        {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
        {"role": "user", "content": answers},
    ])
    
    return grade["correct"]


def relevance(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
    """
    Evaluator for RAG answer helpfulness and relevance.
    Checks if the answer addresses the question without requiring a reference answer.
    
    Args:
        inputs: Input dictionary with "question" key
        outputs: Output dictionary with "answer" key
        
    Returns:
        True if the answer is relevant, False otherwise
    """
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    
    grade = relevance_llm.invoke([
        {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
        {"role": "user", "content": answer},
    ])
    
    return grade["relevant"]


def groundedness(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
    """
    Evaluator for RAG answer groundedness.
    Checks if the answer is based on the retrieved documents and doesn't hallucinate.
    
    Args:
        inputs: Input dictionary with "question" key
        outputs: Output dictionary with "answer" and "documents" keys
        
    Returns:
        True if the answer is grounded, False otherwise
    """
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    
    grade = grounded_llm.invoke([
        {"role": "system", "content": GROUNDED_INSTRUCTIONS},
        {"role": "user", "content": answer},
    ])
    
    return grade["grounded"]


def retrieval_relevance(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
    """
    Evaluator for document relevance.
    Checks if the retrieved documents are relevant to the question.
    
    Args:
        inputs: Input dictionary with "question" key
        outputs: Output dictionary with "documents" key
        
    Returns:
        True if the documents are relevant, False otherwise
    """
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
        {"role": "user", "content": answer},
    ])
    
    return grade["relevant"]


def create_evaluation_dataset(
    dataset_name: str = "HR Chatbot Q&A",
    examples: List[Dict[str, Any]] = None,
    overwrite: bool = False
) -> str:
    """
    Create or get a LangSmith dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset
        examples: List of examples with "inputs" and "outputs" keys
        overwrite: If True, overwrite existing dataset with new examples
        
    Returns:
        Dataset name (for use in evaluation)
    """
    if examples is None:
        # Default examples - you should replace these with your actual HR policy questions
        examples = [
            {
                "inputs": {"question": "What is the notice period for employees?"},
                "outputs": {"answer": "The notice period varies by grade and employment type. Please refer to the HR policy document for specific details."},
            },
            {
                "inputs": {"question": "What are the leave policies?"},
                "outputs": {"answer": "Leave policies include annual leave, sick leave, and other types. Specific details can be found in the HR policy document."},
            },
        ]
    
    # Check if dataset exists
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples
        )
        logger.info(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    else:
        if overwrite:
            # Delete existing dataset and create new one
            existing_dataset = client.read_dataset(dataset_name=dataset_name)
            client.delete_dataset(dataset_id=existing_dataset.id)
            dataset = client.create_dataset(dataset_name=dataset_name)
            client.create_examples(
                dataset_id=dataset.id,
                examples=examples
            )
            logger.info(f"Overwritten dataset '{dataset_name}' with {len(examples)} examples")
        else:
            logger.info(f"Dataset '{dataset_name}' already exists - using existing dataset")
    
    return dataset_name


def run_evaluation(
    dataset_name: str = None,
    experiment_prefix: str = "hr-chatbot-rag-eval",
    examples: List[Dict[str, Any]] = None
):
    """
    Run evaluation on the HR chatbot.
    
    Args:
        dataset_name: Name of the dataset (if None, will create default)
        experiment_prefix: Prefix for the experiment name in LangSmith
        examples: List of examples for the dataset (if None, uses defaults)
        
    Returns:
        Evaluation results
    """
    # Create or get dataset
    # If examples are provided, create/update the dataset with those examples
    # If no examples and no dataset_name, create default dataset
    if examples is not None:
        # Examples provided - create/update dataset with these examples
        if dataset_name is None:
            dataset_name = "HR Chatbot Q&A"  # Default name
        dataset_name = create_evaluation_dataset(dataset_name=dataset_name, examples=examples)
    elif dataset_name is None:
        # No examples and no dataset name - create default dataset
        dataset_name = create_evaluation_dataset(examples=examples)
    # else: dataset_name provided but no examples - use existing dataset
    
    logger.info(f"Starting evaluation with dataset: {dataset_name}")
    
    # Run evaluation
    experiment_results = client.evaluate(
        hr_chatbot_wrapper,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix=experiment_prefix,
        metadata={
            "version": "HR Chatbot RAG Evaluation",
            "chatbot_model": _hr_config_manager.get(ConfigKeys.MODEL_NAME) or settings.CHAT_MODEL,
            "chatbot_temperature": _hr_config_manager.get(ConfigKeys.MODEL_TEMPERATURE) or settings.CHAT_MODEL_TEMPERATURE,
            "grader_model": grader_model_name,
            "grader_temperature": grader_temperature,
        },
    )
    
    logger.info("Evaluation completed successfully!")
    # Log experiment name if available
    if hasattr(experiment_results, 'experiment_name'):
        logger.info(f"Experiment name: {experiment_results.experiment_name}")
    elif hasattr(experiment_results, 'experiment_url'):
        logger.info(f"Results available in LangSmith: {experiment_results.experiment_url}")
    else:
        logger.info("Results available in LangSmith. Check your LangSmith dashboard for details.")
    
    # Try to convert to pandas if available
    try:
        import pandas as pd
        df = experiment_results.to_pandas()
        logger.info("\nEvaluation Summary:")
        logger.info(f"\n{df.to_string()}")
        return experiment_results, df
    except ImportError:
        logger.info("pandas not available, skipping DataFrame conversion")
        return experiment_results, None


if __name__ == "__main__":
    """
    Example usage:
    
    # With default examples
    python evaluations/hr_chatbot/evaluate_hr_chatbot.py
    
    # With existing dataset name
    python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-name "My Dataset"
    
    # With JSON file containing examples
    python evaluations/hr_chatbot/evaluate_hr_chatbot.py --dataset-file my_examples.json
    
    # With custom examples programmatically
    from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation
    
    custom_examples = [
        {
            "inputs": {"question": "What is the notice period for grade 8 employees?"},
            "outputs": {"answer": "The notice period for grade 8 employees is 30 days."},
        },
        # Add more examples...
    ]
    
    run_evaluation(
        dataset_name="HR Chatbot Custom Evaluation",
        examples=custom_examples
    )
    """
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate HR Chatbot using LangSmith")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset to use or create (existing LangSmith dataset name)"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to JSON file containing examples (alternative to --dataset-name)"
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="hr-chatbot-rag-eval",
        help="Prefix for the experiment name in LangSmith"
    )
    
    args = parser.parse_args()
    
    # Load examples from file if provided
    examples = None
    if args.dataset_file:
        try:
            with open(args.dataset_file, "r") as f:
                examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from {args.dataset_file}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {args.dataset_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {e}")
            raise
    
    # Run evaluation
    results, df = run_evaluation(
        dataset_name=args.dataset_name,
        experiment_prefix=args.experiment_prefix,
        examples=examples
    )
    
    if df is not None:
        print("\n" + "="*80)
        print("Evaluation Results Summary")
        print("="*80)
        print(df.to_string())
        print("\n" + "="*80)
        # Try to get experiment URL or name
        if hasattr(results, 'experiment_url'):
            print(f"View full results in LangSmith: {results.experiment_url}")
        elif hasattr(results, 'experiment_name'):
            print(f"Experiment name: {results.experiment_name}")
            print("View full results in your LangSmith dashboard")
        else:
            print("View full results in your LangSmith dashboard")
        print("="*80)

