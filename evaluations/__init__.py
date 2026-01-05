"""
Evaluations Module

This module contains evaluation scripts and utilities for evaluating chatbots.

Structure:
- core/          - Generic evaluation framework (reusable for any chatbot)
- hr_chatbot/    - HR chatbot-specific evaluation scripts and datasets
"""

from evaluations.core import (
    ChatbotEvaluator,
    create_evaluation_dataset as create_eval_dataset,
    run_evaluation as run_eval,
)

__all__ = [
    "ChatbotEvaluator",
    "create_eval_dataset",
    "run_eval",
]
