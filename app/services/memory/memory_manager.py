"""
Memory Management for LangChain Agents
Handles trim and summarize operations for chat history
"""
import sys
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logging import logger
from app.core.memory_config import MemoryConfig, MemoryStrategy
from app.infra.llm.llm_manager import get_llm_manager


class MemoryManager:
    """
    Manages memory operations (trim, summarize) for chatbot conversations.
    """
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self._summarize_llm = None
        if config.summarize_model:
            self._summarize_llm = get_llm_manager().get_llm(model_name=config.summarize_model)
    
    def process_messages(
        self,
        messages: List[BaseMessage],
        thread_id: str
    ) -> List[BaseMessage]:
        """
        Process messages based on memory strategy.
        
        Args:
            messages: List of messages in the conversation
            thread_id: Thread identifier (for logging)
            
        Returns:
            Processed list of messages
        """
        if self.config.strategy == MemoryStrategy.NONE:
            return messages
        
        # Count non-system messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        message_count = len(non_system_messages)
        
        # Apply strategy
        if self.config.strategy == MemoryStrategy.TRIM:
            return self._trim_messages(messages, system_messages, non_system_messages, message_count)
        elif self.config.strategy == MemoryStrategy.SUMMARIZE:
            return self._summarize_messages(messages, system_messages, non_system_messages, message_count)
        elif self.config.strategy == MemoryStrategy.TRIM_AND_SUMMARIZE:
            return self._trim_and_summarize_messages(messages, system_messages, non_system_messages, message_count)
        
        return messages
    
    def _trim_messages(
        self,
        messages: List[BaseMessage],
        system_messages: List[BaseMessage],
        non_system_messages: List[BaseMessage],
        message_count: int
    ) -> List[BaseMessage]:
        """
        Trim messages to keep only the most recent ones based on trim_keep_messages.
        
        Args:
            messages: Original list of messages
            system_messages: System messages (always kept)
            non_system_messages: Non-system messages
            message_count: Count of non-system messages
            
        Returns:
            Trimmed list of messages
        """
        if message_count <= self.config.trim_keep_messages:
            return messages
        
        # Keep only the most recent messages
        trimmed = non_system_messages[-self.config.trim_keep_messages:]
        
        logger.debug(
            f"Trimmed messages: {message_count} -> {len(trimmed)} "
            f"(keeping last {self.config.trim_keep_messages})"
        )
        
        return system_messages + trimmed
    
    def _summarize_messages(
        self,
        messages: List[BaseMessage],
        system_messages: List[BaseMessage],
        non_system_messages: List[BaseMessage],
        message_count: int
    ) -> List[BaseMessage]:
        """
        Summarize old messages when threshold is reached, keep recent ones.
        
        Args:
            messages: Original list of messages
            system_messages: System messages (always kept)
            non_system_messages: Non-system messages
            message_count: Count of non-system messages
            
        Returns:
            Messages with old ones summarized
        """
        if message_count <= self.config.summarize_threshold:
            return messages
        
        # Split into old and recent messages
        # Keep recent messages (up to trim_keep_messages)
        split_point = len(non_system_messages) - self.config.trim_keep_messages
        old_messages = non_system_messages[:split_point]
        recent_messages = non_system_messages[split_point:]
        
        # Summarize old messages
        summary = self._create_summary(old_messages)
        summary_message = SystemMessage(
            content=f"Previous conversation summary: {summary}"
        )
        
        logger.debug(
            f"Summarized {len(old_messages)} messages into summary, "
            f"keeping {len(recent_messages)} recent messages"
        )
        
        return system_messages + [summary_message] + recent_messages
    
    def _trim_and_summarize_messages(
        self,
        messages: List[BaseMessage],
        system_messages: List[BaseMessage],
        non_system_messages: List[BaseMessage],
        message_count: int
    ) -> List[BaseMessage]:
        """
        Trim messages and summarize old ones.
        
        Args:
            messages: Original list of messages
            system_messages: System messages (always kept)
            non_system_messages: Non-system messages
            message_count: Count of non-system messages
            
        Returns:
            Processed messages
        """
        if message_count <= self.config.summarize_threshold:
            # Just trim if below threshold
            if message_count > self.config.trim_keep_messages:
                return system_messages + non_system_messages[-self.config.trim_keep_messages:]
            return messages
        
        # Split messages
        split_point = len(non_system_messages) - self.config.trim_keep_messages
        old_messages = non_system_messages[:split_point]
        recent_messages = non_system_messages[split_point:]
        
        # Summarize old messages
        summary = self._create_summary(old_messages)
        summary_message = SystemMessage(
            content=f"Previous conversation summary: {summary}"
        )
        
        logger.debug(
            f"Trimmed and summarized: {len(old_messages)} old messages summarized, "
            f"{len(recent_messages)} recent messages kept"
        )
        
        return system_messages + [summary_message] + recent_messages
    
    def _create_summary(self, messages: List[BaseMessage]) -> str:
        """
        Create a summary of old messages using LLM.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            Summary text
        """
        try:
            # Get LLM for summarization
            if self._summarize_llm:
                llm = self._summarize_llm
                model_name = self.config.summarize_model or "custom model"
            else:
                llm = get_llm_manager().get_llm()
                model_name = "default chat model"
            
            logger.info(f"Using {model_name} for summarization")
            
            # Convert messages to text
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages
            ])
            
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of the following conversation. 
Focus on key topics, decisions, and important information that should be remembered for future context.

Conversation:
{conversation_text}

Summary:"""
            
            # Generate summary
            response = llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated summary using {model_name} (length: {len(summary)} chars, from {len(messages)} messages)")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            # Fallback: return a simple summary
            return f"Previous conversation with {len(messages)} messages (summary generation failed)"
