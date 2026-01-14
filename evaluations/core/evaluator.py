"""
Generic Chatbot Evaluator - LLM-as-Judge Evaluation Framework

This module provides a reusable evaluation framework that can be used
to evaluate any chatbot implementation that follows the ChatbotAgent interface.

Based on: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from uuid import uuid4

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langsmith import Client, traceable
from langchain_core.documents import Document

# Handle TypedDict and Annotated imports
try:
    from typing_extensions import TypedDict, Annotated
except ImportError:
    from typing import TypedDict, Annotated

from src.shared.config.logging import logger
from src.shared.config.langsmith import initialize_langsmith
from src.domain.retrieval.service import RetrievalService
from src.infrastructure.vectorstore.manager import get_vector_store
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys
from src.infrastructure.llm.manager import get_llm_manager


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


class ScannabilityGrade(TypedDict):
    """Schema for scannability evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    scannable: Annotated[
        bool,
        ...,
        "True if the answer uses bold headers and bullet points effectively, False otherwise",
    ]


SCANNABILITY_INSTRUCTIONS = """You are a UX Auditor.
STUDENT ANSWER: {student_answer}

Grade Criteria:
(1) Scannability: Does the answer use **bold headers** to separate categories of info?
(2) Bullet Points: Are factual details presented as bullet points rather than sentences? 
(3) Directness: Does the answer provide the core fact in the very first sentence? (FAIL if it starts with filler).

Scannable: [True/False]

Reasoning: Evaluate if the user can find the specific answer in under 3 seconds. Check for headers, bullets, and the absence of dense text."""

CORRECTNESS_INSTRUCTIONS = """You are an HR Compliance Auditor.
QUESTION: {question}
GROUND TRUTH: {ground_truth}
STUDENT ANSWER: {student_answer}

Grade Criteria:
(1) Factual Core: Do the specific numbers, dates, or rules match the Ground Truth?
(2) Hallucination: Penalize only if the student includes facts that contradict the Ground Truth or materially change its meaning. Do not penalize supplemental, non-contradictory details not present in the Ground Truth.
(3) Brevity Tolerance: Do NOT penalize the student for omitting peripheral details (like "how to apply" or "manager approval") if they were not explicitly asked for, even if they are in the Ground Truth.
(4) Safety: Did the student admit if info was missing?

Correctness: [True/False]
Reasoning: Compare the core facts. If the student provides the correct 'Limit' but skips the 'Process' (to remain concise), mark as CORRECT."""

RELEVANCE_INSTRUCTIONS = """You are a Teacher grading for Conciseness and Intent.
QUESTION: {question}
STUDENT ANSWER: {student_answer}

Grade Criteria:
(1) Intent Alignment: Does the answer directly address the specific user intent in the first sentence?
(2) Contextual Buffer: Do NOT penalize for including "Eligibility", "Approvals", or "Limits" if they are directly related to the question (e.g., if asked about a benefit, knowing the limit is relevant).
(3) Noise Filter: FAIL the answer ONLY if it contains "Policy Dumping"—defined as including entirely different topics (e.g., explaining 'Sick Leave' when asked about 'Maternity Leave', or listing 'How to Apply' when only asked 'How many days').

Relevance: [True/False]

Reasoning: Did the bot answer the question immediately? Is the additional information provided useful context for this specific question, or is it a generic dump of the whole document?"""

GROUNDED_INSTRUCTIONS = """You are a Citation Auditor.
FACTS: {context}
STUDENT ANSWER: {student_answer}

Grade Criteria:
(1) 1:1 Mapping: Exactly ONE unique number per unique filename in 'Sources'.
(2) Attribution: Does every bullet point end with [n]?
(3) Body Cleanliness: Are there ANY filenames (e.g., .pdf) or file paths in the body text? (FAIL if yes).
(4) Support: Are all claims found in the FACTS?

Grounded: [True/False]
Reasoning: Check for citation consistency and ensure no "policy.pdf" text leaked into the response body."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are an HR Data Auditor. 
QUESTION: {question}
FACTS: {context}

Grade Criteria:
(1) Topical Alignment: Do the FACTS belong to the same HR category as the QUESTION?
(2) Signal Presence: If the question asks for a 'Formula' and the FACTS contain 'F&F Process' but no formula, mark as TRUE for retrieval (the system found the right document).
(3) Failure: ONLY mark as False if the FACTS are completely unrelated (e.g., user asks about 'Salary' but snippets are about 'Harassment').
Relevance: [True/False]

Reasoning: Identify the specific keywords in the QUESTION and confirm if the retrieved FACTS contain the corresponding values or rules."""


def _extract_documents_from_agent_result(result: Any) -> List[Document]:
    """
    Extract retrieved documents from agent execution result.
    
    The agent's tool calls with response_format="content_and_artifact" store
    the artifact in ToolMessage objects. The artifact is typically in the
    response_metadata or as a structured attribute.
    
    Args:
        result: Agent invocation result (dict with "messages" key)
        
    Returns:
        List of Document objects, or empty list if extraction fails
    """
    try:
        from langchain_core.messages import ToolMessage
        
        if not isinstance(result, dict) or "messages" not in result:
            return []
        
        documents = []
        messages = result["messages"]
        
        # Look for ToolMessage objects that contain artifacts from retrieval tool
        for message in messages:
            # Check if it's a ToolMessage (from tool execution)
            is_tool_message = (
                isinstance(message, ToolMessage) or
                (isinstance(message, dict) and message.get("type") == "tool")
            )
            
            if not is_tool_message:
                continue
            
            # Try multiple ways to extract the artifact
            
            # Method 1: Check response_metadata for artifact
            response_metadata = None
            if hasattr(message, 'response_metadata'):
                response_metadata = message.response_metadata
            elif isinstance(message, dict):
                response_metadata = message.get('response_metadata', {})
            
            if response_metadata and 'artifact' in response_metadata:
                artifact = response_metadata['artifact']
                if isinstance(artifact, list) and artifact:
                    if isinstance(artifact[0], dict) and 'content' in artifact[0]:
                        for doc_data in artifact:
                            documents.append(
                                Document(
                                    page_content=doc_data.get('content', ''),
                                    metadata=doc_data.get('metadata', {})
                                )
                            )
                        if documents:
                            logger.debug(f"Extracted {len(documents)} documents from response_metadata")
                            return documents
            
            # Method 2: Check if content is the artifact (list of dicts)
            content = message.content if hasattr(message, 'content') else message.get('content', '')
            
            if isinstance(content, list) and content:
                # Check if it looks like an artifact (list of dicts with 'content' and 'metadata')
                if isinstance(content[0], dict) and 'content' in content[0]:
                    for doc_data in content:
                        documents.append(
                            Document(
                                page_content=doc_data.get('content', ''),
                                metadata=doc_data.get('metadata', {})
                            )
                        )
                    if documents:
                        logger.debug(f"Extracted {len(documents)} documents from content (list)")
                        return documents
            
            # Method 3: Try parsing content as JSON string
            if isinstance(content, str) and content.strip().startswith('['):
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, list) and parsed:
                        if isinstance(parsed[0], dict) and 'content' in parsed[0]:
                            for doc_data in parsed:
                                documents.append(
                                    Document(
                                        page_content=doc_data.get('content', ''),
                                        metadata=doc_data.get('metadata', {})
                                    )
                                )
                            if documents:
                                logger.debug(f"Extracted {len(documents)} documents from JSON content")
                                return documents
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass
            
            # Method 4: Check for artifact attribute directly
            if hasattr(message, 'artifact'):
                artifact = message.artifact
                if isinstance(artifact, list) and artifact:
                    if isinstance(artifact[0], dict) and 'content' in artifact[0]:
                        for doc_data in artifact:
                            documents.append(
                                Document(
                                    page_content=doc_data.get('content', ''),
                                    metadata=doc_data.get('metadata', {})
                                )
                            )
                        if documents:
                            logger.debug(f"Extracted {len(documents)} documents from artifact attribute")
                            return documents
        
        # If we couldn't extract from tool messages, return empty list
        logger.debug("Could not extract documents from agent execution, will retrieve separately")
        return []
        
    except Exception as e:
        logger.warning(f"Error extracting documents from agent result: {e}")
        return []


class ChatbotEvaluator:
    """
    Generic chatbot evaluator that works with any ChatbotAgent implementation.
    
    This class provides a reusable evaluation framework that can evaluate
    any chatbot following the ChatbotAgent interface.
    """
    
    def __init__(
        self,
        chatbot_getter: Callable,
        chatbot_type: str,
        config_filename: Optional[str] = None,
        config_manager: Optional[ChatbotConfigManager] = None,
        retrieval_k: int = 6,
        grader_model_name: Optional[str] = None,
        grader_temperature: Optional[float] = None,
    ):
        """
        Initialize the chatbot evaluator.
        
        Args:
            chatbot_getter: Function that returns a chatbot instance (e.g., get_hr_chatbot)
            chatbot_type: Type identifier for the chatbot (e.g., "hr")
            config_filename: Optional YAML config filename (e.g., "hr_chatbot_config.yaml")
            config_manager: Optional ChatbotConfigManager instance (if None, will create from config_filename)
            retrieval_k: Number of documents to retrieve for fallback retrieval (default: 6)
            grader_model_name: Optional model name for graders (if None, uses chatbot config)
            grader_temperature: Optional temperature for graders (if None, uses 0 for deterministic evaluation)
        """
        self.chatbot_getter = chatbot_getter
        self.chatbot_type = chatbot_type
        self.retrieval_k = retrieval_k
        
        # Initialize LangSmith
        initialize_langsmith(force_enable=True)
        self.client = Client()
        
        # Setup config manager
        if config_manager is not None:
            self.config_manager = config_manager
        elif config_filename is not None:
            self.config_manager = ChatbotConfigManager(config_filename)
        else:
            self.config_manager = None
        
        # Store grader model name for metadata
        self.grader_model_name = grader_model_name
        
        # Setup grader LLM
        self._setup_grader_llm(grader_model_name, grader_temperature)
        
        # Cache chatbot instance and services
        self._chatbot_instance = None
        self._vector_store = None
        self._retrieval_service = None
    
    def _setup_grader_llm(
        self,
        grader_model_name: Optional[str] = None,
        grader_temperature: Optional[float] = None
    ):
        """Setup grader LLMs with structured output."""
        from src.shared.config.settings import settings
        
        # Get model configuration
        if grader_model_name is None:
            if self.config_manager is not None:
                grader_model_name = self.config_manager.get(ConfigKeys.MODEL_NAME) or settings.CHAT_MODEL
            else:
                grader_model_name = settings.CHAT_MODEL
        
        # Store the resolved model name
        self.grader_model_name = grader_model_name
        
        if grader_temperature is None:
            if self.config_manager is not None:
                grader_temperature = self.config_manager.get(ConfigKeys.MODEL_TEMPERATURE)
                if grader_temperature is not None:
                    grader_temperature = float(grader_temperature)
                else:
                    grader_temperature = 0
            else:
                grader_temperature = 0  # Use 0 for evaluation (more deterministic)
        
        logger.info(f"Using model for graders: {grader_model_name} (temperature: {grader_temperature})")
        
        # Use LLM manager to create the grader LLM
        llm_manager = get_llm_manager()
        grader_llm = llm_manager.get_llm(
            model_name=grader_model_name,
            temperature=grader_temperature,
            max_tokens=None,
            use_cache=False
        )
        
        # Create structured output graders
        try:
            self.correctness_llm = grader_llm.with_structured_output(
                CorrectnessGrade, method="json_schema", strict=True
            )
            self.relevance_llm = grader_llm.with_structured_output(
                RelevanceGrade, method="json_schema", strict=True
            )
            self.grounded_llm = grader_llm.with_structured_output(
                GroundedGrade, method="json_schema", strict=True
            )
            self.retrieval_relevance_llm = grader_llm.with_structured_output(
                RetrievalRelevanceGrade, method="json_schema", strict=True
            )
            self.scannability_llm = grader_llm.with_structured_output(
                ScannabilityGrade, method="json_schema", strict=True
            )
            logger.info("Successfully created structured output graders")
        except Exception as e:
            logger.warning(f"Failed to create structured output graders: {e}")
            logger.info("Falling back to regular LLM calls (may need manual parsing)")
            self.correctness_llm = self.relevance_llm = self.grounded_llm = self.retrieval_relevance_llm = self.scannability_llm = grader_llm
    
    def _get_chatbot_instance(self):
        """Get or create chatbot instance (singleton pattern for evaluation)."""
        if self._chatbot_instance is None:
            self._chatbot_instance = self.chatbot_getter()
            logger.info(f"Initialized {self.chatbot_type} chatbot instance for evaluation (will be reused)")
        return self._chatbot_instance
    
    def _get_retrieval_service(self):
        """Get or create retrieval service (singleton pattern for evaluation)."""
        if self._retrieval_service is None:
            self._vector_store = get_vector_store(self.chatbot_type)
            self._retrieval_service = RetrievalService(self._vector_store)
        return self._retrieval_service
    
    @traceable()
    def chatbot_wrapper(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper function for chatbot that returns both answer and retrieved documents.
        This is needed for evaluation as we need access to both the response and the documents.
        
        Args:
            inputs: Dictionary with "question" key
            
        Returns:
            Dictionary with "answer" and "documents" keys
        """
        question = inputs["question"]
        
        # Get chatbot instance (reused from cache)
        chatbot = self._get_chatbot_instance()
        
        # Generate a unique thread_id for this evaluation run
        thread_id = f"eval_{uuid4().hex[:8]}"
        
        # Call chat() method to get the answer (fully relies on chat() function)
        # This ensures evaluation results match direct execution behavior
        answer = chatbot.chat(
            query=question,
            thread_id=thread_id,
            user_id="evaluation_user"
        )
        
        # Extract documents from checkpointer state (same documents used by chat())
        # This ensures we evaluate against the exact documents the agent retrieved
        documents = []
        try:
            from src.infrastructure.storage.checkpointing.manager import get_checkpointer_manager
            checkpointer_manager = get_checkpointer_manager()
            config = checkpointer_manager.get_config(thread_id, "evaluation_user")
            checkpoint = checkpointer_manager.checkpointer.get(config)
            
            if checkpoint and "channel_values" in checkpoint:
                messages = checkpoint["channel_values"].get("messages", [])
                documents = _extract_documents_from_agent_result({"messages": messages})
        except Exception as e:
            logger.debug(f"Could not extract documents from checkpointer: {e}")
        
        # Fallback to direct retrieval if extraction from checkpointer failed
        if not documents:
            logger.debug("Falling back to direct retrieval for documents")
            retrieval_service = self._get_retrieval_service()
            _, artifact = retrieval_service.retrieve(query=question, k=self.retrieval_k)
            
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
    
    def correctness(self, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> bool:
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
        grade = self.correctness_llm.invoke([
            {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
            {"role": "user", "content": answers},
        ])
        
        return grade["correct"]
    
    def relevance(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
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
        
        grade = self.relevance_llm.invoke([
            {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
            {"role": "user", "content": answer},
        ])
        
        return grade["relevant"]
    
    def groundedness(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
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
        
        grade = self.grounded_llm.invoke([
            {"role": "system", "content": GROUNDED_INSTRUCTIONS},
            {"role": "user", "content": answer},
        ])
        
        return grade["grounded"]
    
    def retrieval_relevance(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
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
        grade = self.retrieval_relevance_llm.invoke([
            {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
            {"role": "user", "content": answer},
        ])
        
        return grade["relevant"]
    
    def scannability(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """
        Evaluator for answer scannability and visual structure.
        Checks if the answer uses bold headers and bullet points effectively.
        
        Args:
            inputs: Input dictionary with "question" key
            outputs: Output dictionary with "answer" key
            
        Returns:
            True if the answer is scannable, False otherwise
        """
        answer = f"STUDENT ANSWER: {outputs['answer']}"
        
        grade = self.scannability_llm.invoke([
            {"role": "system", "content": SCANNABILITY_INSTRUCTIONS},
            {"role": "user", "content": answer},
        ])
        
        return grade["scannable"]
    
    def create_evaluation_dataset(
        self,
        dataset_name: str,
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
            examples = []
        
        # Check if dataset exists
        if not self.client.has_dataset(dataset_name=dataset_name):
            dataset = self.client.create_dataset(dataset_name=dataset_name)
            if examples:
                self.client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
            logger.info(f"Created dataset '{dataset_name}' with {len(examples)} examples")
        else:
            if overwrite:
                # Delete existing dataset and create new one
                existing_dataset = self.client.read_dataset(dataset_name=dataset_name)
                self.client.delete_dataset(dataset_id=existing_dataset.id)
                dataset = self.client.create_dataset(dataset_name=dataset_name)
                if examples:
                    self.client.create_examples(
                        dataset_id=dataset.id,
                        examples=examples
                    )
                logger.info(f"Overwritten dataset '{dataset_name}' with {len(examples)} examples")
            else:
                logger.info(f"Dataset '{dataset_name}' already exists - using existing dataset")
        
        return dataset_name
    
    def run_evaluation(
        self,
        dataset_name: str = None,
        experiment_prefix: str = None,
        examples: List[Dict[str, Any]] = None,
        evaluators: Optional[List[Callable]] = None
    ):
        """
        Run evaluation on the chatbot.
        
        Args:
            dataset_name: Name of the dataset (if None, will create default)
            experiment_prefix: Prefix for the experiment name in LangSmith
            examples: List of examples for the dataset (if None, uses existing dataset)
            evaluators: Optional list of custom evaluator functions (if None, uses default evaluators)
            
        Returns:
            Evaluation results
        """
        from src.shared.config.settings import settings
        
        # Default experiment prefix
        if experiment_prefix is None:
            experiment_prefix = f"{self.chatbot_type}-chatbot-rag-eval"
        
        # Create or get dataset
        if examples is not None:
            # Examples provided - create/update dataset with these examples
            if dataset_name is None:
                dataset_name = f"{self.chatbot_type.capitalize()} Chatbot Q&A"
            dataset_name = self.create_evaluation_dataset(
                dataset_name=dataset_name,
                examples=examples,
                overwrite=True
            )
        elif dataset_name is None:
            # No examples and no dataset name - create empty dataset
            dataset_name = f"{self.chatbot_type.capitalize()} Chatbot Q&A"
            dataset_name = self.create_evaluation_dataset(
                dataset_name=dataset_name,
                examples=examples
            )
        
        logger.info(f"Starting evaluation with dataset: {dataset_name}")
        
        # Use default evaluators if not provided
        if evaluators is None:
            evaluators = [
                self.correctness,
                self.groundedness,
                self.relevance,
                self.retrieval_relevance,
                self.scannability
            ]
        
        # Get model configuration for metadata
        if self.config_manager is not None:
            chatbot_model = self.config_manager.get(ConfigKeys.MODEL_NAME) or settings.CHAT_MODEL
            chatbot_temperature = self.config_manager.get(ConfigKeys.MODEL_TEMPERATURE) or settings.CHAT_MODEL_TEMPERATURE
        else:
            chatbot_model = settings.CHAT_MODEL
            chatbot_temperature = settings.CHAT_MODEL_TEMPERATURE
        
        # Run evaluation
        experiment_results = self.client.evaluate(
            self.chatbot_wrapper,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata={
                "version": f"{self.chatbot_type.capitalize()} Chatbot RAG Evaluation",
                "chatbot_type": self.chatbot_type,
                "chatbot_model": chatbot_model,
                "chatbot_temperature": chatbot_temperature,
                "grader_model": self.grader_model_name,
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


# Convenience functions for backward compatibility
def create_evaluation_dataset(
    dataset_name: str,
    examples: List[Dict[str, Any]] = None,
    overwrite: bool = False
) -> str:
    """
    Standalone function to create a LangSmith dataset.
    
    This is a convenience function that creates a temporary evaluator
    just to access the dataset creation functionality.
    """
    initialize_langsmith(force_enable=True)
    client = Client()
    
    if examples is None:
        examples = []
    
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        if examples:
            client.create_examples(
                dataset_id=dataset.id,
                examples=examples
            )
        logger.info(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    else:
        if overwrite:
            existing_dataset = client.read_dataset(dataset_name=dataset_name)
            client.delete_dataset(dataset_id=existing_dataset.id)
            dataset = client.create_dataset(dataset_name=dataset_name)
            if examples:
                client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
            logger.info(f"Overwritten dataset '{dataset_name}' with {len(examples)} examples")
        else:
            logger.info(f"Dataset '{dataset_name}' already exists - using existing dataset")
    
    return dataset_name


def run_evaluation(
    chatbot_getter: Callable,
    chatbot_type: str,
    dataset_name: str = None,
    experiment_prefix: str = None,
    examples: List[Dict[str, Any]] = None,
    config_filename: Optional[str] = None,
    **kwargs
):
    """
    Standalone function to run evaluation on a chatbot.
    
    This is a convenience function that creates an evaluator and runs evaluation.
    
    Args:
        chatbot_getter: Function that returns a chatbot instance
        chatbot_type: Type identifier for the chatbot
        dataset_name: Name of the dataset
        experiment_prefix: Prefix for the experiment name
        examples: List of examples for the dataset
        config_filename: Optional YAML config filename
        **kwargs: Additional arguments to pass to ChatbotEvaluator
        
    Returns:
        Evaluation results
    """
    evaluator = ChatbotEvaluator(
        chatbot_getter=chatbot_getter,
        chatbot_type=chatbot_type,
        config_filename=config_filename,
        **kwargs
    )
    
    return evaluator.run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        examples=examples
    )

