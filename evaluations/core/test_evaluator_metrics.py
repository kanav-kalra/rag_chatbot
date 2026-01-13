"""
Test Runner for Evaluation Metrics

This script validates that all evaluation metrics work correctly by running
test cases against the evaluator. It can be used to:
1. Validate that evaluators are working correctly
2. Test edge cases for each metric
3. Debug evaluation issues

Usage:
    python evaluations/core/test_evaluator_metrics.py [--metric METRIC_NAME] [--verbose]
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluations.core.evaluator import ChatbotEvaluator
from evaluations.core.evaluation_test_cases import (
    get_test_cases_for_metric,
    get_all_test_cases,
    get_test_case_summary
)
from src.shared.config.logging import logger


def create_mock_chatbot_getter():
    """Create a mock chatbot getter for testing."""
    def mock_getter():
        # Return a mock chatbot that we won't actually use
        # since we're testing the evaluators directly
        return None
    return mock_getter


def test_metric(
    evaluator: ChatbotEvaluator,
    metric_name: str,
    test_cases: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test a specific metric with its test cases.
    
    Args:
        evaluator: ChatbotEvaluator instance
        metric_name: Name of the metric to test
        test_cases: List of test cases for the metric
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with test results
    """
    results = {
        "metric": metric_name,
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }
    
    # Get the evaluator method
    evaluator_method = getattr(evaluator, metric_name, None)
    if evaluator_method is None:
        logger.error(f"Evaluator method '{metric_name}' not found")
        results["errors"] = len(test_cases)
        return results
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {metric_name.upper()} metric ({len(test_cases)} test cases)")
    logger.info(f"{'='*60}")
    
    for i, test_case in enumerate(test_cases, 1):
        test_name = test_case.get("name", f"test_{i}")
        expected_score = test_case.get("expected_score", None)
        
        try:
            # Run the evaluator
            if metric_name == "correctness":
                # Correctness needs reference_outputs
                score = evaluator_method(
                    test_case["inputs"],
                    test_case["outputs"],
                    test_case.get("reference_outputs", {})
                )
            else:
                # Other metrics only need inputs and outputs
                score = evaluator_method(
                    test_case["inputs"],
                    test_case["outputs"]
                )
            
            # Check if result matches expectation
            passed = (score == expected_score) if expected_score is not None else True
            
            if passed:
                results["passed"] += 1
                status = "✓ PASS"
            else:
                results["failed"] += 1
                status = "✗ FAIL"
            
            detail = {
                "name": test_name,
                "expected": expected_score,
                "actual": score,
                "passed": passed,
                "description": test_case.get("description", "")
            }
            results["details"].append(detail)
            
            if verbose or not passed:
                logger.info(f"\n{status} - {test_name}")
                logger.info(f"  Description: {test_case.get('description', 'N/A')}")
                logger.info(f"  Expected: {expected_score}, Actual: {score}")
                if not passed:
                    logger.warning(f"  ⚠ Mismatch! Expected {expected_score} but got {score}")
            elif verbose:
                logger.info(f"  {status} - {test_name}")
                
        except Exception as e:
            results["errors"] += 1
            detail = {
                "name": test_name,
                "expected": expected_score,
                "actual": None,
                "passed": False,
                "error": str(e),
                "description": test_case.get("description", "")
            }
            results["details"].append(detail)
            logger.error(f"\n✗ ERROR - {test_name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    logger.info(f"\n{metric_name.upper()} Summary:")
    logger.info(f"  Total: {results['total']}")
    logger.info(f"  Passed: {results['passed']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Errors: {results['errors']}")
    logger.info(f"  Success Rate: {results['passed']/results['total']*100:.1f}%")
    
    return results


def run_all_tests(
    evaluator: ChatbotEvaluator,
    metrics: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run tests for all or specified metrics.
    
    Args:
        evaluator: ChatbotEvaluator instance
        metrics: List of metric names to test (None = all)
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with all test results
    """
    all_test_cases = get_all_test_cases()
    
    if metrics is None:
        metrics = list(all_test_cases.keys())
    
    all_results = {
        "total_metrics": len(metrics),
        "total_tests": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_errors": 0,
        "metrics": {}
    }
    
    for metric_name in metrics:
        if metric_name not in all_test_cases:
            logger.warning(f"Metric '{metric_name}' not found. Available: {list(all_test_cases.keys())}")
            continue
        
        test_cases = all_test_cases[metric_name]
        results = test_metric(evaluator, metric_name, test_cases, verbose)
        
        all_results["metrics"][metric_name] = results
        all_results["total_tests"] += results["total"]
        all_results["total_passed"] += results["passed"]
        all_results["total_failed"] += results["failed"]
        all_results["total_errors"] += results["errors"]
    
    # Print overall summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Metrics Tested: {all_results['total_metrics']}")
    logger.info(f"Total Test Cases: {all_results['total_tests']}")
    logger.info(f"Passed: {all_results['total_passed']}")
    logger.info(f"Failed: {all_results['total_failed']}")
    logger.info(f"Errors: {all_results['total_errors']}")
    if all_results['total_tests'] > 0:
        success_rate = all_results['total_passed'] / all_results['total_tests'] * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
    
    return all_results


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test evaluation metrics with predefined test cases"
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Test a specific metric (correctness, relevance, groundedness, retrieval_relevance, scannability)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each test case"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test cases and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        summary = get_test_case_summary()
        print("\nAvailable Test Cases:")
        print("=" * 50)
        for metric, count in summary.items():
            print(f"{metric.upper()}: {count} test cases")
        print(f"\nTotal: {sum(summary.values())} test cases")
        return
    
    # Create evaluator (we won't actually use the chatbot, just the evaluators)
    logger.info("Initializing evaluator...")
    evaluator = ChatbotEvaluator(
        chatbot_getter=create_mock_chatbot_getter(),
        chatbot_type="test",
        config_filename=None
    )
    
    # Run tests
    metrics_to_test = [args.metric] if args.metric else None
    results = run_all_tests(evaluator, metrics_to_test, args.verbose)
    
    # Exit with error code if there are failures
    if results["total_failed"] > 0 or results["total_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

