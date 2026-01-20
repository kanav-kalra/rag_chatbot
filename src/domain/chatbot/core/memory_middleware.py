"""
Memory Middleware Factory for Chatbot Agents
Handles creation of memory management middleware based on configuration.
Follows SOLID principles:
- Single Responsibility: Only responsible for creating memory middleware
- Open/Closed: Extensible through subclassing without modification
- Dependency Inversion: Depends on abstractions (MemoryConfig, LLM manager)
"""
import sys
from pathlib import Path
from typing import List, Any, Optional, Callable

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage, SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from src.shared.config.logging import logger
from src.shared.memory.config import MemoryConfig, MemoryStrategy
from src.infrastructure.llm.manager import get_llm_manager


class MemoryMiddlewareFactory:
    """
    Factory for creating memory management middleware based on configuration.
    
    This class encapsulates all memory-related middleware creation logic,
    following the Single Responsibility Principle. It creates middleware
    functions that can be used with LangChain's create_agent.
    """
    
    def __init__(self, memory_config: MemoryConfig, default_model_name: str, summary_prompt_template: Optional[str] = None):
        """
        Initialize memory middleware factory.
        
        Args:
            memory_config: Memory configuration specifying strategy and parameters
            default_model_name: Default model name to use for summarization if not specified
            summary_prompt_template: Optional custom prompt template for summarization. 
                                    Should contain {old_summary} and {conversation_text} placeholders.
                                    If None, uses default prompt.
        """
        self.memory_config = memory_config
        self.default_model_name = default_model_name
        self.summary_prompt_template = summary_prompt_template
    
    def create_middleware(self) -> List[Callable]:
        """
        Create memory management middleware based on memory configuration.
        
        Returns:
            List of middleware functions to apply to the agent
        """
        middleware = []
        
        if self.memory_config.strategy == MemoryStrategy.NONE:
            return middleware
        
        # Create appropriate middleware based on strategy
        if self.memory_config.strategy == MemoryStrategy.TRIM:
            trim_middleware = self._create_trim_middleware()
            middleware.append(trim_middleware)
        elif self.memory_config.strategy == MemoryStrategy.SUMMARIZE:
            summarize_middleware = self._create_summarize_middleware()
            middleware.append(summarize_middleware)
        elif self.memory_config.strategy == MemoryStrategy.TRIM_AND_SUMMARIZE:
            # Combined middleware: summarize before trim_keep_messages, then keep trim_keep_messages
            combined_middleware = self._create_trim_and_summarize_middleware()
            middleware.append(combined_middleware)
        
        return middleware
    
    def _should_skip_tool_calling_phase(self, messages: List[Any]) -> bool:
        """
        Check if we're in the tool-calling phase and should skip memory operations.
        
        Args:
            messages: List of messages from state
            
        Returns:
            True if we should skip (in tool-calling phase), False otherwise
        """
        if not messages:
            return False
        
        last_message = messages[-1]
        # Check if last message is a ToolMessage (tool results received)
        if isinstance(last_message, ToolMessage):
            logger.debug("Skipping memory operation: in tool-calling phase (ToolMessage detected)")
            return True
        # Check if last message is an AIMessage with tool_calls (about to process tool results)
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.debug("Skipping memory operation: in tool-calling phase (AIMessage with tool_calls detected)")
            return True
        
        return False
    
    def _trim_messages_logic(self, messages: List[Any], trim_keep_messages: int) -> dict[str, Any] | None:
        """
        Core logic for trimming messages.
        
        Args:
            messages: List of messages to trim
            trim_keep_messages: Number of recent messages to keep
            
        Returns:
            Dictionary with new messages or None if no trimming needed
        """
        # Calculate total messages to keep: first message (if exists) + last N messages
        total_keep = trim_keep_messages + (1 if messages and messages[0] not in messages[-trim_keep_messages:] else 0)
        
        if len(messages) <= total_keep:
            return None  # No changes needed
        
        first_msg = messages[0] if messages else None
        recent_messages = messages[-trim_keep_messages:]
        
        # Combine first message with recent messages, avoiding duplication
        if first_msg and first_msg not in recent_messages:
            new_messages = [first_msg] + recent_messages
        else:
            new_messages = recent_messages
        
        logger.debug(
            f"Trimmed messages: {len(messages)} -> {len(new_messages)} "
            f"(keeping first message + last {trim_keep_messages} messages)"
        )
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    def _find_summarize_split_point(
        self, 
        non_system_messages: List[Any], 
        summarize_threshold: int,
        trim_keep_messages: Optional[int] = None
    ) -> Optional[int]:
        """
        Find the split point for summarization.
        
        Args:
            non_system_messages: List of non-system messages
            summarize_threshold: Threshold for summarization
            trim_keep_messages: Optional trim limit (for trim_and_summarize strategy)
            
        Returns:
            Split point index or None if no split needed
        """
        message_count = len(non_system_messages)
        
        # Find the last user message to preserve complete conversation turns
        # Gemini requires function calls to come immediately after user messages
        last_user_msg_idx = None
        for i in range(len(non_system_messages) - 1, -1, -1):
            if isinstance(non_system_messages[i], HumanMessage):
                last_user_msg_idx = i
                break
        
        # Calculate split point
        if trim_keep_messages is not None:
            # For trim_and_summarize: summarize everything before trim_keep_messages
            min_split = message_count - trim_keep_messages
            if last_user_msg_idx is not None and last_user_msg_idx < min_split:
                split_point = last_user_msg_idx
            else:
                split_point = max(0, min_split)
        else:
            # For summarize only: summarize everything before the threshold
            split_point = max(0, message_count - summarize_threshold)
            # If we have a user message before split point, start from there to preserve sequence
            if last_user_msg_idx is not None and last_user_msg_idx < split_point:
                split_point = last_user_msg_idx
        
        # Safety check: ensure we have messages to summarize
        if split_point <= 0 or split_point >= len(non_system_messages):
            return None  # Not enough messages to summarize
        
        return split_point
    
    def _create_summary_and_replace_messages(
        self,
        old_messages: List[Any],
        recent_messages: List[Any],
        system_messages: List[SystemMessage],
        summary_generator: 'SummaryGenerator',
        trim_keep_messages: Optional[int] = None
    ) -> dict[str, Any] | None:
        """
        Create summary and build new message list.
        
        Args:
            old_messages: Messages to summarize
            recent_messages: Messages to keep
            system_messages: System messages to preserve
            summary_generator: Summary generator instance
            trim_keep_messages: Optional trim limit for logging
            
        Returns:
            Dictionary with new messages or None on error
        """
        # Extract existing summary if present
        old_summary = SummaryGenerator.extract_old_summary(system_messages)
        
        # Filter out old summary from system messages (we'll add new one)
        summary_prefix = "Previous conversation summary: "
        filtered_system_messages = [
            msg for msg in system_messages
            if not (isinstance(msg, SystemMessage) and msg.content.startswith(summary_prefix))
        ]
        
        # Summarize old messages (with old summary if it exists)
        try:
            summary = summary_generator.create_summary(old_messages, old_summary=old_summary)
            summary_message = SystemMessage(
                content=f"Previous conversation summary: {summary}"
            )
            
            trim_info = f" (trimmed to {trim_keep_messages})" if trim_keep_messages else " (no trimming)"
            logger.debug(
                f"Summarized {len(old_messages)} messages into summary, "
                f"keeping {len(recent_messages)} recent messages{trim_info}"
                f"{', incorporating old summary' if old_summary else ''}"
            )
            
            new_messages = filtered_system_messages + [summary_message] + recent_messages
            
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *new_messages
                ]
            }
        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            return None
    
    def _create_trim_middleware(self) -> Callable:
        """
        Create a @before_model middleware function for trimming messages.
        
        Returns:
            Middleware function decorated with @before_model
        """
        # Ensure minimum value of 2 to keep at least first message + 1 recent message
        trim_keep_messages = max(2, self.memory_config.trim_keep_messages)
        
        @before_model
        def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            """
            Trim messages to keep only the most recent ones.
            Keeps the first message (usually system message) and the last N messages.
            Based on LangChain's trim messages pattern.
            """
            messages = state.get("messages", [])
            return self._trim_messages_logic(messages, trim_keep_messages)
        
        return trim_messages
    
    def _create_summarize_middleware(self) -> Callable:
        """
        Create a @before_model middleware function for summarizing messages.
        Only summarizes, does not trim messages.
        
        Returns:
            Middleware function decorated with @before_model
        """
        # Ensure minimum value: summarize_threshold >= 2
        summarize_threshold = max(2, self.memory_config.summarize_threshold)
        summarize_model = self.memory_config.summarize_model or self.default_model_name
        
        # Create summary generator function
        summary_generator = SummaryGenerator(summarize_model, self.summary_prompt_template)
        
        @before_model
        def summarize_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            """
            Summarize old messages when threshold is reached, keep all recent messages.
            Preserves conversation sequence for Gemini compatibility.
            
            Note: This middleware runs before each model invocation. In agent workflows,
            there are typically two model calls: one to decide tools, and one to generate
            the final response. We only summarize before the first call (when last message
            is a HumanMessage), not during tool-calling phase.
            """
            messages = state.get("messages", [])
            
            # Skip summarization if we're in the tool-calling phase
            if self._should_skip_tool_calling_phase(messages):
                return None
            
            # Count non-system messages
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            message_count = len(non_system_messages)
            
            if message_count <= summarize_threshold:
                return None  # No changes needed
            
            # Find split point for summarization
            split_point = self._find_summarize_split_point(non_system_messages, summarize_threshold)
            if split_point is None:
                return None  # Not enough messages to summarize
            
            old_messages = non_system_messages[:split_point]
            recent_messages = non_system_messages[split_point:]
            
            # Create summary and replace messages
            return self._create_summary_and_replace_messages(
                old_messages, recent_messages, system_messages, summary_generator
            )
        
        return summarize_messages
    
    def _create_trim_and_summarize_middleware(self) -> Callable:
        """
        Create a @before_model middleware function that combines trim and summarize.
        Summarizes everything before trim_keep_messages, then keeps trim_keep_messages recent messages.
        
        Returns:
            Middleware function decorated with @before_model
        """
        # Ensure minimum values: summarize_threshold >= 2, trim_keep_messages >= 2
        summarize_threshold = max(2, self.memory_config.summarize_threshold)
        trim_keep_messages = max(2, self.memory_config.trim_keep_messages)
        summarize_model = self.memory_config.summarize_model or self.default_model_name
        
        # Create summary generator function
        summary_generator = SummaryGenerator(summarize_model, self.summary_prompt_template)
        
        @before_model
        def trim_and_summarize_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            """
            Summarize messages before trim_keep_messages, then keep trim_keep_messages recent messages.
            Preserves conversation sequence for Gemini compatibility.
            
            Note: This middleware runs before each model invocation. In agent workflows,
            there are typically two model calls: one to decide tools, and one to generate
            the final response. We only summarize before the first call (when last message
            is a HumanMessage), not during tool-calling phase.
            """
            messages = state.get("messages", [])
            
            # Skip summarization if we're in the tool-calling phase
            if self._should_skip_tool_calling_phase(messages):
                return None
            
            # Count non-system messages
            system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
            non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            message_count = len(non_system_messages)
            
            # First, check if we need to summarize (based on threshold)
            if message_count <= summarize_threshold:
                # Not enough messages to summarize, just trim if needed
                return self._trim_messages_logic(messages, trim_keep_messages)
            
            # Find split point for summarization (using trim_keep_messages limit)
            split_point = self._find_summarize_split_point(
                non_system_messages, summarize_threshold, trim_keep_messages
            )
            if split_point is None:
                return None  # Not enough messages to summarize
            
            old_messages = non_system_messages[:split_point]
            recent_messages = non_system_messages[split_point:]
            
            # Create summary and replace messages
            result = self._create_summary_and_replace_messages(
                old_messages, recent_messages, system_messages, summary_generator, trim_keep_messages
            )
            
            # Fallback to trim if summarization fails
            if result is None:
                logger.warning("Summarization failed, falling back to trim")
                return self._trim_messages_logic(messages, trim_keep_messages)
            
            return result
        
        return trim_and_summarize_messages


class SummaryGenerator:
    """
    Handles generation of conversation summaries using LLM.
    
    Follows Single Responsibility Principle - only responsible for
    creating summaries from message lists.
    """
    
    def __init__(self, model_name: str, summary_prompt_template: Optional[str] = None):
        """
        Initialize summary generator.
        
        Args:
            model_name: Model name to use for summarization
            summary_prompt_template: Optional custom prompt template for summarization.
                                    Should contain {old_summary} and {conversation_text} placeholders.
                                    If None, uses default prompt.
        """
        self.model_name = model_name
        self.summary_prompt_template = summary_prompt_template
    
    @staticmethod
    def extract_old_summary(system_messages: List[SystemMessage]) -> Optional[str]:
        """
        Extract existing summary from system messages if present.
        
        Args:
            system_messages: List of system messages to check
            
        Returns:
            Existing summary text if found, None otherwise
        """
        summary_prefix = "Previous conversation summary: "
        for msg in system_messages:
            if isinstance(msg, SystemMessage) and msg.content.startswith(summary_prefix):
                # Extract the summary text (remove the prefix)
                return msg.content[len(summary_prefix):].strip()
        return None
    
    def create_summary(self, messages: List[Any], old_summary: Optional[str] = None) -> str:
        """
        Create a summary of old messages using LLM.
        If an old summary exists, it will be combined with new messages to create a final summary.
        
        Args:
            messages: List of messages to summarize
            old_summary: Optional previous summary to incorporate into the new summary
            
        Returns:
            Summary text
        """
        try:
            # Get LLM for summarization
            summarize_llm = get_llm_manager().get_llm(model_name=self.model_name)
            
            logger.info(f"Using {self.model_name} for summarization")
            
            # Convert messages to text
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages
            ])
            
            # Create summarization prompt
            if self.summary_prompt_template:
                # Use custom prompt template
                # Handle case where old_summary might be None
                old_summary_text = old_summary if old_summary else ""
                prompt = self.summary_prompt_template.format(
                    old_summary=old_summary_text,
                    conversation_text=conversation_text
                )
            elif old_summary:
                # Default prompt with old summary
                prompt = f"""Please provide a concise summary that combines the previous conversation summary with the new conversation messages.
Focus on key topics, decisions, and important information that should be remembered for future context.
Merge the old summary with new information to create a comprehensive final summary.

Previous Summary:
{old_summary}

New Conversation Messages:
{conversation_text}

Combined Summary:"""
            else:
                # Default prompt without old summary
                prompt = f"""Please provide a concise summary of the following conversation. 
Focus on key topics, decisions, and important information that should be remembered for future context.

Conversation:
{conversation_text}

Summary:"""
            
            # Generate summary
            response = summarize_llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(
                f"Generated summary using {self.model_name} "
                f"(length: {len(summary)} chars, from {len(messages)} messages"
                f"{', with old summary' if old_summary else ''})"
            )
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            # Fallback: return a simple summary
            return f"Previous conversation with {len(messages)} messages (summary generation failed)"
