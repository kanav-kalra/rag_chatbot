"""
Comprehensive Evaluation Test Cases for All Metrics

This module contains test cases for evaluating all metrics:
- Correctness: Factual accuracy against ground truth
- Relevance: Answer helpfulness and relevance to question
- Groundedness: Answer based on retrieved documents with proper citations
- Retrieval Relevance: Retrieved documents relevant to question
- Scannability: Visual structure with headers and bullet points

Each test case includes:
- Input question
- Expected output (for correctness)
- Sample answer (student answer)
- Retrieved documents (for groundedness and retrieval relevance)
- Expected metric scores
"""

from typing import Dict, List, Any
from langchain_core.documents import Document


# ============================================================================
# CORRECTNESS TEST CASES
# ============================================================================

CORRECTNESS_TEST_CASES = [
    {
        "name": "correctness_positive_exact_match",
        "inputs": {"question": "What is the staff welfare budget per person per quarter?"},
        "outputs": {
            "answer": "The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June."
        },
        "reference_outputs": {
            "answer": "The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June."
        },
        "expected_score": True,
        "description": "Exact match with ground truth - should pass"
    },
    {
        "name": "correctness_positive_semantic_match",
        "inputs": {"question": "How long can a Performance Improvement Plan last?"},
        "outputs": {
            "answer": "A Performance Improvement Plan typically lasts between 1 month and 2 months, depending on the specific case as per the defined process."
        },
        "reference_outputs": {
            "answer": "A Performance Improvement Plan generally lasts between 1 month and 2 months, depending on the case as per the defined process."
        },
        "expected_score": True,
        "description": "Semantic match with minor wording differences - should pass"
    },
    {
        "name": "correctness_negative_wrong_number",
        "inputs": {"question": "What is the staff welfare budget per person per quarter?"},
        "outputs": {
            "answer": "The staff welfare budget is INR 2,000 per person per quarter."
        },
        "reference_outputs": {
            "answer": "The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June."
        },
        "expected_score": False,
        "description": "Wrong number (2000 vs 1500) - should fail"
    },
    {
        "name": "correctness_negative_missing_info",
        "inputs": {"question": "What is the staff welfare budget per person per quarter?"},
        "outputs": {
            "answer": "The staff welfare budget is INR 1,500 per person per quarter."
        },
        "reference_outputs": {
            "answer": "The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June."
        },
        "expected_score": False,
        "description": "Missing important context (financial year) - should fail"
    },
    {
        "name": "correctness_positive_admits_gap",
        "inputs": {"question": "What is the maximum number of team members for a staff welfare activity?"},
        "outputs": {
            "answer": "I don't have information about the maximum number of team members for a staff welfare activity in the available documents."
        },
        "reference_outputs": {
            "answer": "Information not available in policy documents."
        },
        "expected_score": True,
        "description": "Correctly admits information gap - should pass"
    },
    {
        "name": "correctness_negative_hallucinates",
        "inputs": {"question": "What is the maximum number of team members for a staff welfare activity?"},
        "outputs": {
            "answer": "The maximum number of team members for a staff welfare activity is 50 people."
        },
        "reference_outputs": {
            "answer": "Information not available in policy documents."
        },
        "expected_score": False,
        "description": "Hallucinates answer when info is missing - should fail"
    },
]


# ============================================================================
# RELEVANCE TEST CASES
# ============================================================================

RELEVANCE_TEST_CASES = [
    {
        "name": "relevance_positive_direct_answer",
        "inputs": {"question": "What is the process to get approval for using the staff welfare budget?"},
        "outputs": {
            "answer": """**Approval Process for Staff Welfare Budget**

To get approval for using the staff welfare budget:

1. **Manager Planning**: Manager at grade 7 or above plans the engagement activity
2. **HOD Approval**: Seek approval from the HOD at grade 10 and above
3. **HRBP Approval**: Share participant list and event date with HRBP before the event
4. **Reimbursement**: After the event, submit original bills with HOD and HRBP approvals to Finance"""
        },
        "expected_score": True,
        "description": "Directly addresses the question with actionable steps - should pass"
    },
    {
        "name": "relevance_positive_complete_leave_info",
        "inputs": {"question": "What are the leave policies for employees?"},
        "outputs": {
            "answer": """**Leave Policies**

**Eligibility**: All full-time employees are eligible for leave benefits.

**Annual Leave**: 20 days per year
**Sick Leave**: 10 days per year  
**Casual Leave**: 5 days per year

**Process**: Submit leave requests through the HR portal at least 3 days in advance."""
        },
        "expected_score": True,
        "description": "Covers eligibility, duration, and process for leave question - should pass"
    },
    {
        "name": "relevance_negative_off_topic",
        "inputs": {"question": "What is the process to get approval for using the staff welfare budget?"},
        "outputs": {
            "answer": "The company provides health insurance benefits including medical, dental, and vision coverage. Employees can enroll during the open enrollment period."
        },
        "expected_score": False,
        "description": "Completely off-topic (benefits vs approval process) - should fail"
    },
    {
        "name": "relevance_negative_incomplete",
        "inputs": {"question": "What are the leave policies for employees?"},
        "outputs": {
            "answer": "Employees get 20 days of annual leave per year."
        },
        "expected_score": False,
        "description": "Incomplete - missing eligibility, other leave types, and process - should fail"
    },
    {
        "name": "relevance_negative_vague",
        "inputs": {"question": "What is the process to get approval for using the staff welfare budget?"},
        "outputs": {
            "answer": "You need to get approval from your manager and HR. Then submit the bills for reimbursement."
        },
        "expected_score": False,
        "description": "Too vague, missing specific steps and requirements - should fail"
    },
]


# ============================================================================
# GROUNDEDNESS TEST CASES
# ============================================================================

GROUNDEDNESS_TEST_CASES = [
    {
        "name": "groundedness_positive_proper_citations",
        "inputs": {"question": "What is the staff welfare budget?"},
        "outputs": {
            "answer": """**Staff Welfare Budget**

The staff welfare budget is INR 1,500 per person per quarter [1]. This budget is available for eligible team engagement activities within the financial year July to June [1].

**Sources:**
[1] staff_welfare_policy.pdf""",
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "All claims supported by facts with proper citations - should pass"
    },
    {
        "name": "groundedness_positive_multiple_sources",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "answer": """**Leave Policies**

**Annual Leave**: Full-time employees are entitled to 20 days of annual leave per year [1].

**Sick Leave**: Employees receive 10 days of sick leave annually [1].

**Casual Leave**: 5 days of casual leave are available per year [2].

**Sources:**
[1] leave_policy.pdf
[2] employee_handbook.pdf""",
            "documents": [
                Document(
                    page_content="Full-time employees are entitled to 20 days of annual leave per year. Employees receive 10 days of sick leave annually.",
                    metadata={"source": "leave_policy.pdf"}
                ),
                Document(
                    page_content="5 days of casual leave are available per year for all employees.",
                    metadata={"source": "employee_handbook.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "Multiple sources with proper citations and deduplication - should pass"
    },
    {
        "name": "groundedness_negative_hallucination",
        "inputs": {"question": "What is the staff welfare budget?"},
        "outputs": {
            "answer": """**Staff Welfare Budget**

The staff welfare budget is INR 2,500 per person per quarter [1]. This budget can be used for any team activity throughout the year [1].

**Sources:**
[1] staff_welfare_policy.pdf""",
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Hallucinated numbers (2500 vs 1500) and wrong info (any activity vs eligible) - should fail"
    },
    {
        "name": "groundedness_negative_no_citations",
        "inputs": {"question": "What is the staff welfare budget?"},
        "outputs": {
            "answer": """**Staff Welfare Budget**

The staff welfare budget is INR 1,500 per person per quarter. This budget is available for eligible team engagement activities within the financial year July to June.

**Sources:**
[1] staff_welfare_policy.pdf""",
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "No citation markers in body text - should fail"
    },
    {
        "name": "groundedness_negative_filename_in_body",
        "inputs": {"question": "What is the staff welfare budget?"},
        "outputs": {
            "answer": """**Staff Welfare Budget**

According to staff_welfare_policy.pdf, the staff welfare budget is INR 1,500 per person per quarter [1].

**Sources:**
[1] staff_welfare_policy.pdf""",
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Filename in body text (staff_welfare_policy.pdf) - should fail"
    },
    {
        "name": "groundedness_negative_duplicate_sources",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "answer": """**Leave Policies**

Employees get 20 days of annual leave [1] and 10 days of sick leave [2].

**Sources:**
[1] leave_policy.pdf
[2] leave_policy.pdf""",
            "documents": [
                Document(
                    page_content="Full-time employees are entitled to 20 days of annual leave per year.",
                    metadata={"source": "leave_policy.pdf"}
                ),
                Document(
                    page_content="Employees receive 10 days of sick leave annually.",
                    metadata={"source": "leave_policy.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Duplicate filename in Sources section - should fail"
    },
    {
        "name": "groundedness_negative_unsupported_claim",
        "inputs": {"question": "What is the staff welfare budget?"},
        "outputs": {
            "answer": """**Staff Welfare Budget**

The staff welfare budget is INR 1,500 per person per quarter [1]. Managers can approve up to INR 5,000 for special events [1].

**Sources:**
[1] staff_welfare_policy.pdf""",
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Unsupported claim (INR 5,000 for special events) not in facts - should fail"
    },
]


# ============================================================================
# RETRIEVAL RELEVANCE TEST CASES
# ============================================================================

RETRIEVAL_RELEVANCE_TEST_CASES = [
    {
        "name": "retrieval_relevance_positive_exact_match",
        "inputs": {"question": "What is the staff welfare budget per person per quarter?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="The staff welfare budget is INR 1,500 per person per quarter for eligible team engagement activities within the financial year July to June.",
                    metadata={"source": "staff_welfare_policy.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "Exact keyword match (staff welfare budget) - should pass"
    },
    {
        "name": "retrieval_relevance_positive_semantic_match",
        "inputs": {"question": "How much time off can I take for vacation?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="Full-time employees are entitled to 20 days of annual leave per year. This includes vacation time and personal days.",
                    metadata={"source": "leave_policy.pdf"}
                ),
                Document(
                    page_content="Employees can accrue up to 30 days of PTO (Paid Time Off) which can be used for vacation, sick leave, or personal matters.",
                    metadata={"source": "pto_policy.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "Semantic match (time off/vacation -> annual leave/PTO) - should pass"
    },
    {
        "name": "retrieval_relevance_positive_category_match",
        "inputs": {"question": "What are the health insurance benefits?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="The company provides comprehensive health insurance coverage including medical, dental, and vision benefits. Employees can enroll during open enrollment period.",
                    metadata={"source": "benefits_policy.pdf"}
                ),
                Document(
                    page_content="Wellness programs and EAP (Employee Assistance Program) services are available to all employees.",
                    metadata={"source": "wellness_program.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "Category match (Benefits: insurance, wellness, EAP) - should pass"
    },
    {
        "name": "retrieval_relevance_positive_with_noise",
        "inputs": {"question": "What is the salary structure for grade 8 employees?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="Grade 8 employees receive a base salary ranging from INR 800,000 to INR 1,200,000 per annum. Performance bonuses are calculated separately.",
                    metadata={"source": "compensation_policy.pdf"}
                ),
                Document(
                    page_content="Fire safety procedures must be followed in all office buildings. Emergency exits are clearly marked.",
                    metadata={"source": "safety_manual.pdf"}
                )
            ]
        },
        "expected_score": True,
        "description": "Relevant document with some noise (fire safety) - should pass (low bar)"
    },
    {
        "name": "retrieval_relevance_negative_completely_unrelated",
        "inputs": {"question": "What is the salary structure for grade 8 employees?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="Fire safety procedures must be followed in all office buildings. Emergency exits are clearly marked on each floor.",
                    metadata={"source": "safety_manual.pdf"}
                ),
                Document(
                    page_content="The cafeteria serves vegetarian and non-vegetarian meals. Operating hours are 9 AM to 3 PM.",
                    metadata={"source": "cafeteria_info.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Completely unrelated (fire safety, cafeteria vs salary) - should fail"
    },
    {
        "name": "retrieval_relevance_negative_wrong_category",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "documents": [
                Document(
                    page_content="The company provides comprehensive health insurance coverage including medical, dental, and vision benefits.",
                    metadata={"source": "benefits_policy.pdf"}
                ),
                Document(
                    page_content="401k retirement plans are available with company matching up to 5% of salary.",
                    metadata={"source": "retirement_benefits.pdf"}
                )
            ]
        },
        "expected_score": False,
        "description": "Wrong category (Benefits vs Leave) - should fail"
    },
]


# ============================================================================
# SCANNABILITY TEST CASES
# ============================================================================

SCANNABILITY_TEST_CASES = [
    {
        "name": "scannability_positive_headers_and_bullets",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "answer": """**Leave Policies**

**Annual Leave**
- Full-time employees: 20 days per year
- Accrual: 1.67 days per month
- Carryover: Up to 5 days to next year

**Sick Leave**
- 10 days per year
- No carryover allowed
- Requires medical certificate for 3+ consecutive days

**Casual Leave**
- 5 days per year
- Cannot be combined with other leave types"""
        },
        "expected_score": True,
        "description": "Uses bold headers and bullet points effectively - should pass"
    },
    {
        "name": "scannability_positive_step_by_step",
        "inputs": {"question": "What is the approval process for staff welfare budget?"},
        "outputs": {
            "answer": """**Approval Process for Staff Welfare Budget**

**Step 1: Planning**
- Manager at grade 7+ plans the activity
- Determine participant list and event date

**Step 2: HOD Approval**
- Seek approval from HOD at grade 10+
- Submit activity proposal

**Step 3: HRBP Approval**
- Share participant list and event date with HRBP
- Get approval before the event

**Step 4: Reimbursement**
- Submit original bills after the event
- Include HOD and HRBP approvals
- Submit to Finance for processing"""
        },
        "expected_score": True,
        "description": "Clear headers with numbered steps and bullets - should pass"
    },
    {
        "name": "scannability_negative_dense_paragraph",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "answer": """The leave policies for employees include annual leave, sick leave, and casual leave. Full-time employees are entitled to 20 days of annual leave per year which accrues at 1.67 days per month. Up to 5 days can be carried over to the next year. Sick leave is 10 days per year with no carryover allowed and requires a medical certificate for 3 or more consecutive days. Casual leave is 5 days per year and cannot be combined with other leave types. All leave requests must be submitted through the HR portal at least 3 days in advance."""
        },
        "expected_score": False,
        "description": "Dense paragraph with multiple facts - should fail"
    },
    {
        "name": "scannability_negative_no_headers",
        "inputs": {"question": "What is the approval process?"},
        "outputs": {
            "answer": """To get approval for staff welfare budget, first the manager at grade 7 or above plans the engagement activity. Then they seek approval from the HOD at grade 10 and above. After that, the details including participant list and event date must be shared with the HRBP for approval before the event. Finally, after the event, original bills along with HOD and HRBP approvals are submitted to Finance for reimbursement."""
        },
        "expected_score": False,
        "description": "No bold headers, just plain text - should fail"
    },
    {
        "name": "scannability_negative_no_bullets",
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {
            "answer": """**Leave Policies**

**Annual Leave**
Full-time employees are entitled to 20 days of annual leave per year. This accrues at 1.67 days per month. Up to 5 days can be carried over to the next year.

**Sick Leave**
Employees receive 10 days of sick leave annually. No carryover is allowed. A medical certificate is required for 3 or more consecutive days."""
        },
        "expected_score": False,
        "description": "Has headers but uses paragraphs instead of bullets for details - should fail"
    },
    {
        "name": "scannability_negative_mixed_format",
        "inputs": {"question": "What is the approval process?"},
        "outputs": {
            "answer": """**Approval Process**

The manager at grade 7 or above plans the engagement activity and seeks approval from the HOD at grade 10 and above. Then the details including the participant list and event date must be shared with the HRBP for approval before the event. After the event, original bills along with HOD and HRBP approvals are submitted to Finance for reimbursement. The process requires careful documentation and timely submission of all required materials."""
        },
        "expected_score": False,
        "description": "Has header but uses dense paragraph for steps - should fail"
    },
]


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

ALL_TEST_CASES = {
    "correctness": CORRECTNESS_TEST_CASES,
    "relevance": RELEVANCE_TEST_CASES,
    "groundedness": GROUNDEDNESS_TEST_CASES,
    "retrieval_relevance": RETRIEVAL_RELEVANCE_TEST_CASES,
    "scannability": SCANNABILITY_TEST_CASES,
}


def get_test_cases_for_metric(metric_name: str) -> List[Dict[str, Any]]:
    """
    Get all test cases for a specific metric.
    
    Args:
        metric_name: Name of the metric (correctness, relevance, groundedness, retrieval_relevance, scannability)
        
    Returns:
        List of test cases for the metric
    """
    return ALL_TEST_CASES.get(metric_name.lower(), [])


def get_all_test_cases() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all test cases organized by metric.
    
    Returns:
        Dictionary mapping metric names to their test cases
    """
    return ALL_TEST_CASES


def get_test_case_summary() -> Dict[str, int]:
    """
    Get a summary of test cases count per metric.
    
    Returns:
        Dictionary mapping metric names to test case counts
    """
    return {
        metric: len(cases) 
        for metric, cases in ALL_TEST_CASES.items()
    }


if __name__ == "__main__":
    # Print summary of test cases
    print("Evaluation Test Cases Summary")
    print("=" * 50)
    summary = get_test_case_summary()
    for metric, count in summary.items():
        print(f"{metric.upper()}: {count} test cases")
    print(f"\nTotal: {sum(summary.values())} test cases")
    
    # Print example test case for each metric
    print("\n" + "=" * 50)
    print("Example Test Cases:")
    print("=" * 50)
    
    for metric, cases in ALL_TEST_CASES.items():
        if cases:
            example = cases[0]
            print(f"\n{metric.upper()} - {example['name']}:")
            print(f"  Description: {example['description']}")
            print(f"  Expected Score: {example['expected_score']}")

