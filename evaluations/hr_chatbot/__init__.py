"""
HR Chatbot Evaluation Module

This module contains evaluation scripts and utilities for evaluating the HR chatbot.

This module uses the generic evaluation framework from evaluations.core
and provides HR-specific evaluation functions and datasets.
"""

from evaluations.hr_chatbot.evaluate_hr_chatbot import (
    create_evaluation_dataset,
    run_evaluation,
)

__all__ = [
    "create_evaluation_dataset",
    "run_evaluation",
]
