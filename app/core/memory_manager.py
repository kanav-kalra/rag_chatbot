"""
Memory Management Middleware for LangChain Agents
Handles trim and summarize operations for chat history
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logger import logger
from app.core.memory_config import MemoryConfig, MemoryStrategy
from app.llm_manager import get_llm


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
            self._summarize_llm = get_llm(model_name=config.summarize_model)
    
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
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        message_count = len(non_system_messages)
        
        if message_count <= self.config.summarize_threshold:
            # No need to process yet
            if self.config.strategy == MemoryStrategy.TRIM:
                # Still trim if we have more than keep limit
                if message_count > self.config.trim_keep_messages:
                    return self._trim_messages(messages)
            return messages
        
        # Need to process messages
        if self.config.strategy == MemoryStrategy.TRIM:
            return self._trim_messages(messages)
        elif self.config.strategy == MemoryStrategy.SUMMARIZE:
            return self._summarize_messages(messages)
        elif self.config.strategy == MemoryStrategy.TRIM_AND_SUMMARIZE:
            return self._trim_and_summarize_messages(messages)
        
        return messages
    
    def _trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim messages to keep only the most recent ones.
        
        Args:
            messages: List of messages
            
        Returns:
            Trimmed list of messages
        """
        # Keep system messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Keep only the most recent messages
        trimmed = non_system_messages[-self.config.trim_keep_messages:]
        
        logger.debug(
            f"Trimmed messages: {len(non_system_messages)} -> {len(trimmed)} "
            f"(keeping last {self.config.trim_keep_messages})"
        )
        
        return system_messages + trimmed
    
    def _summarize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Summarize old messages and keep recent ones.
        
        Args:
            messages: List of messages
            
        Returns:
            Messages with old ones summarized
        """
        # Keep system messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        if len(non_system_messages) <= self.config.summarize_threshold:
            return messages
        
        # Split into old and recent messages
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
    
    def _trim_and_summarize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim messages and summarize old ones.
        
        Args:
            messages: List of messages
            
        Returns:
            Processed messages
        """
        # First trim to keep more messages than final keep limit
        # This gives us a buffer for summarization
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        if len(non_system_messages) <= self.config.summarize_threshold:
            # Just trim if below threshold
            if len(non_system_messages) > self.config.trim_keep_messages:
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
            llm = self._summarize_llm or get_llm()
            
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
            
            logger.debug(f"Generated summary (length: {len(summary)} chars)")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}", exc_info=True)
            # Fallback: return a simple summary
            return f"Previous conversation with {len(messages)} messages (summary generation failed)"


def create_memory_middleware(memory_config: MemoryConfig):
    """
    Create a memory management middleware for LangChain agents.
    
    Args:
        memory_config: Memory configuration
        
    Returns:
        Middleware function
    """
    memory_manager = MemoryManager(memory_config)
    
    def memory_middleware(state: Dict[str, Any], config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """
        Middleware to process messages based on memory strategy.
        
        Args:
            state: Agent state
            config: Runnable configuration
            
        Returns:
            Updated state or None
        """
        if "messages" not in state:
            return None
        
        messages = state["messages"]
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        # Process messages
        processed_messages = memory_manager.process_messages(messages, thread_id)
        
        if processed_messages != messages:
            return {"messages": processed_messages}
        
        return None
    
    return memory_middleware

