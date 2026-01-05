"""
Generic Evaluation Framework for Chatbots

This module provides a reusable evaluation framework that can be used
to evaluate any chatbot implementation that follows the ChatbotAgent interface.
"""

from evaluations.core.evaluator import (
    ChatbotEvaluator,
    create_evaluation_dataset,
    run_evaluation,
)

__all__ = [
    "ChatbotEvaluator",
    "create_evaluation_dataset",
    "run_evaluation",
]

