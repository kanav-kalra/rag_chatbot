"""
Sample Evaluation Dataset for HR Chatbot

This file contains example evaluation datasets that you can use or modify
for evaluating the HR chatbot. These questions and answers are grounded
in REA India's Mobility Policy, Staff Welfare Policy and the Performance
Improvement Process (Managers Guidelines).
"""

# Example 1: Basic HR Policy Questions
BASIC_HR_DATASET = [
    {
        "inputs": {
            "question": "What is the staff welfare budget per person per quarter?"
        },
        "outputs": {
            "answer": (
                "The staff welfare budget is INR 1,500 per person per quarter "
                "for eligible team engagement activities within the financial year "
                "July to June."
            )
        },
        "TITLE": "Staff welfare budget limit",
    },
    {
        "inputs": {
            "question": "Who can organize activities using the staff welfare budget?"
        },
        "outputs": {
            "answer": (
                "Managers at grade 7 and above can organize team engagement activities "
                "using the staff welfare budget, with prior approval from the HOD at "
                "grade 10 and above."
            )
        },
        "TITLE": "Staff welfare organizer eligibility",
    },
    {
        "inputs": {
            "question": "How long can a Performance Improvement Plan last?"
        },
        "outputs": {
            "answer": (
                "A Performance Improvement Plan generally lasts between 1 month "
                "and 2 months, depending on the case as per the defined process."
            )
        },
        "TITLE": "PIP duration",
    },
    {
        "inputs": {
            "question": "Is counseling required before putting someone on a PIP?"
        },
        "outputs": {
            "answer": (
                "Yes. Before initiating a PIP, the employee should already have "
                "been counseled about improvement areas and observed not improving "
                "over a sufficient period."
            )
        },
        "TITLE": "Pre‑PIP counseling requirement",
    },
    {
        "inputs": {
            "question": "Does the company provide relocation support for internal transfers to another city?"
        },
        "outputs": {
            "answer": (
                "Yes. For movements to a different location within REA India, "
                "relocation support such as movement of household goods, stay "
                "expenses, travel reimbursement and brokerage may be provided, "
                "subject to prior approval and eligibility under the mobility policy."
            )
        },
        "TITLE": "Relocation support for internal moves",
    },
    {
        "inputs": {
            "question": "Is CHRO approval needed for relocation expense claims?"
        },
        "outputs": {
            "answer": (
                "Yes. Prior approval from the Chief Human Resource Officer over "
                "email is required before incurring relocation expenses under "
                "the mobility policy."
            )
        },
        "TITLE": "Relocation CHRO approval",
    },
]

# Example 2: Detailed HR Policy Questions (Policy‑grounded)
DETAILED_HR_DATASET = [
    {
        "inputs": {
            "question": "What is the process to get approval for using the staff welfare budget for a team party?"
        },
        "outputs": {
            "answer": (
                "First, the manager at grade 7 or above plans the engagement activity "
                "and seeks approval from the HOD at grade 10 and above. "
                "Then, the details including the participant list and event date "
                "must be shared with the HRBP for approval before the event. "
                "After the event, original bills along with HOD and HRBP approvals "
                "are submitted to Finance for reimbursement."
            )
        },
        "TITLE": "Staff welfare approval workflow",
    },
    {
        "inputs": {
            "question": "How is unused staff welfare budget treated across quarters and the financial year?"
        },
        "outputs": {
            "answer": (
                "The staff welfare budget is available per person per quarter within "
                "the financial year July to June. Unused budget for a quarter may be "
                "carried forward to the subsequent quarter, but not beyond the second "
                "quarter, and any amount unutilized by the end of the financial year "
                "lapses. Partial under‑utilization in a quarter, such as spending less "
                "than INR 1,500, cannot be carried forward."
            )
        },
        "TITLE": "Staff welfare carry‑forward rules",
    },
    {
        "inputs": {
            "question": "What documentation is required to claim relocation expenses under the mobility policy?"
        },
        "outputs": {
            "answer": (
                "Employees must obtain prior CHRO approval over email before incurring "
                "relocation expenses. Claims are reimbursed on actuals, and original "
                "invoices or receipts must be uploaded in the expense reimbursement tool "
                "along with the approval email."
            )
        },
        "TITLE": "Relocation claim documentation",
    },
    {
        "inputs": {
            "question": "What happens to relocation costs if an employee resigns within 12 months of relocation?"
        },
        "outputs": {
            "answer": (
                "If relocation support has been provided and the employee resigns within "
                "12 months from the assignment commencement date, the relocation expenses "
                "will be recovered from the employee in the Full and Final settlement. "
                "The resignation date is used for this calculation."
            )
        },
        "TITLE": "Relocation cost clawback condition",
    },
    {
        "inputs": {
            "question": "What are the key steps a manager must follow when initiating a Performance Improvement Plan?"
        },
        "outputs": {
            "answer": (
                "The reporting manager must first discuss the PIP case with the reviewing "
                "manager or 2nd‑level supervisor before informing the employee. "
                "A face‑to‑face discussion with the employee is mandatory to communicate "
                "the PIP trigger, expectations, duration and review dates. "
                "The PIP should document specific performance gaps, measurable objectives, "
                "support required and agreed success indicators."
            )
        },
        "TITLE": "PIP initiation steps for managers",
    },
    {
        "inputs": {
            "question": "How should a manager conduct reviews during and at the end of the PIP period?"
        },
        "outputs": {
            "answer": (
                "During the PIP period, the manager must maintain active contact, schedule "
                "frequent review meetings and provide constructive feedback on progress "
                "versus expectations. At the end of the PIP, the manager should analyze "
                "performance with due diligence, communicate the final assessment again
