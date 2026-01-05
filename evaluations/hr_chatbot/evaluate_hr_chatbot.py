"""
HR Chatbot Evaluation - Uses Generic Evaluation Framework

This script evaluates the HR chatbot on four metrics:
1. Correctness: Response vs reference answer
2. Groundedness: Response vs retrieved docs
3. Relevance: Response vs input question
4. Retrieval relevance: Retrieved docs vs input question

This is a thin wrapper around the generic evaluation framework in evaluations.core.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.config.logging import logger
from src.domain.chatbot.hr_chatbot import get_hr_chatbot
from evaluations.core.evaluator import ChatbotEvaluator


# Create HR chatbot evaluator instance
_hr_evaluator = None


def _get_hr_evaluator() -> ChatbotEvaluator:
    """Get or create HR chatbot evaluator instance (singleton pattern)."""
    global _hr_evaluator
    if _hr_evaluator is None:
        _hr_evaluator = ChatbotEvaluator(
            chatbot_getter=get_hr_chatbot,
            chatbot_type="hr",
            config_filename="hr_chatbot_config.yaml",
            retrieval_k=6
        )
        logger.info("Initialized HR chatbot evaluator")
    return _hr_evaluator


def create_evaluation_dataset(
    dataset_name: str = "HR Chatbot Q&A",
    examples: List[Dict[str, Any]] = None,
    overwrite: bool = False
) -> str:
    """
    Create or get a LangSmith dataset for HR chatbot evaluation.
    
    Args:
        dataset_name: Name of the dataset
        examples: List of examples with "inputs" and "outputs" keys
        overwrite: If True, overwrite existing dataset with new examples
        
    Returns:
        Dataset name (for use in evaluation)
    """
    evaluator = _get_hr_evaluator()
    return evaluator.create_evaluation_dataset(
        dataset_name=dataset_name,
        examples=examples,
        overwrite=overwrite
    )


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
        examples: List of examples for the dataset (if None, uses existing dataset)
        
    Returns:
        Evaluation results
    """
    evaluator = _get_hr_evaluator()
    return evaluator.run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        examples=examples
    )


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
