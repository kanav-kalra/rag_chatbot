"""
Sample Evaluation Dataset for HR Chatbot

This file contains example evaluation datasets that you can use or modify
for evaluating the HR chatbot. Replace the examples with your actual HR policy
questions and expected answers.
"""

# Example 1: Basic HR Policy Questions
BASIC_HR_DATASET = [
    {
        "inputs": {"question": "What is the notice period for employees?"},
        "outputs": {"answer": "The notice period varies by grade and employment type. Please refer to the HR policy document for specific details."},
    },
    {
        "inputs": {"question": "What are the leave policies?"},
        "outputs": {"answer": "Leave policies include annual leave, sick leave, and other types. Specific details can be found in the HR policy document."},
    },
    {
        "inputs": {"question": "How do I apply for a leave?"},
        "outputs": {"answer": "You can apply for leave through the HR portal or by submitting a leave application form to your manager and HR department."},
    },
]

# Example 2: Detailed HR Policy Questions (Replace with your actual policies)
DETAILED_HR_DATASET = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days as per company policy."},
    },
    {
        "inputs": {"question": "What are the leave policies for full-time employees?"},
        "outputs": {"answer": "Full-time employees are entitled to 20 days of annual leave, 10 days of sick leave, and 5 days of casual leave per year."},
    },
    {
        "inputs": {"question": "What is the process for requesting a promotion?"},
        "outputs": {"answer": "Employees can request a promotion by discussing with their manager, submitting a formal application through the HR portal, and going through the performance review process."},
    },
    {
        "inputs": {"question": "What benefits are available to employees?"},
        "outputs": {"answer": "Employees are eligible for health insurance, retirement plans, professional development opportunities, and other benefits as outlined in the employee handbook."},
    },
    {
        "inputs": {"question": "What is the company's policy on remote work?"},
        "outputs": {"answer": "Remote work policies vary by department and role. Please consult with your manager and refer to the remote work policy document for specific guidelines."},
    },
]

# Example 3: Edge Cases and Ambiguous Questions
EDGE_CASE_DATASET = [
    {
        "inputs": {"question": "What is the notice period?"},
        "outputs": {"answer": "The notice period depends on the employee's grade and employment type. Please provide your grade or refer to the HR policy document for specific details."},
    },
    {
        "inputs": {"question": "Can I work from home?"},
        "outputs": {"answer": "Remote work policies vary by department and role. Please consult with your manager and refer to the remote work policy document for specific guidelines."},
    },
    {
        "inputs": {"question": "What is the salary for my position?"},
        "outputs": {"answer": "I don't have access to specific salary information. Please contact HR or refer to your employment contract for salary details."},
    },
    {
        "inputs": {"question": "How do I file a complaint?"},
        "outputs": {"answer": "You can file a complaint through the HR portal, by emailing HR directly, or by speaking with your manager. All complaints are handled confidentially."},
    },
]

# Example 4: Complex Multi-part Questions
COMPLEX_HR_DATASET = [
    {
        "inputs": {"question": "What is the notice period for grade 8 employees and what happens if I don't serve it?"},
        "outputs": {"answer": "The notice period for grade 8 employees is 30 days. If you don't serve the full notice period, you may be required to pay compensation or face other consequences as outlined in your employment contract. Please refer to the HR policy for specific details."},
    },
    {
        "inputs": {"question": "What are the leave policies and how do I carry forward unused leave?"},
        "outputs": {"answer": "Leave policies include annual leave, sick leave, and casual leave. Unused annual leave can typically be carried forward up to a certain limit (usually 5-10 days) to the next year, subject to manager approval. Please refer to the leave policy document for specific carry-forward rules."},
    },
]


def get_dataset_by_name(name: str):
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset ("basic", "detailed", "edge_case", "complex")
        
    Returns:
        List of examples for the dataset
    """
    datasets = {
        "basic": BASIC_HR_DATASET,
        "detailed": DETAILED_HR_DATASET,
        "edge_case": EDGE_CASE_DATASET,
        "complex": COMPLEX_HR_DATASET,
    }
    
    return datasets.get(name.lower(), BASIC_HR_DATASET)


if __name__ == "__main__":
    """
    Example usage:
    
    from evaluations.hr_chatbot.sample_evaluation_dataset import get_dataset_by_name
    from evaluations.hr_chatbot.evaluate_hr_chatbot import run_evaluation
    
    # Use a predefined dataset
    examples = get_dataset_by_name("detailed")
    run_evaluation(
        dataset_name="HR Chatbot Detailed Evaluation",
        examples=examples
    )
    
    # Or create your own
    custom_examples = [
        {
            "inputs": {"question": "Your question here"},
            "outputs": {"answer": "Expected answer here"},
        },
        # Add more...
    ]
    run_evaluation(
        dataset_name="HR Chatbot Custom Evaluation",
        examples=custom_examples
    )
    """
    print("Sample evaluation datasets available:")
    print("- BASIC_HR_DATASET: Basic HR policy questions")
    print("- DETAILED_HR_DATASET: Detailed HR policy questions")
    print("- EDGE_CASE_DATASET: Edge cases and ambiguous questions")
    print("- COMPLEX_HR_DATASET: Complex multi-part questions")
    print("\nImport this module and use get_dataset_by_name() to get a dataset.")

