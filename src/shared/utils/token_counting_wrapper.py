"""
Token Counting Observer for Chatbot Agents

This module provides optional token counting and logging functionality for chatbot agents
using the observer pattern. It allows tracking and logging of token usage across different
components of a chat interaction (queries, system prompts, conversation history, retrieved
context, and responses) without duplicating chatbot logic.

Key Components:
--------------
- TokenCountingObserver: Observes chat events and accumulates token counts
- TokenCountingWrapper: Wrapper that manages observer lifecycle and configuration
- Utility Functions: Helper functions for collecting and processing token data

Usage Example:
-------------
```python
from src.shared.utils.token_counting_wrapper import (
    TokenCountingWrapper,
    collect_token_data,
    process_token_counting
)

# Initialize wrapper (typically done in chatbot initialization)
wrapper = TokenCountingWrapper(enabled=True, model_name="gpt-4")

# In chat method:
observer = wrapper.get_observer()
if observer:
    # Collect token data before agent invocation
    token_data = collect_token_data(
        query="What is the leave policy?",
        enhanced_query="[Topic Context: ...] User Question: What is the leave policy?",
        thread_id="thread-123",
        user_id="user-456",
        system_prompt="You are a helpful HR assistant..."
    )
    
    # After agent invocation
    result = agent.invoke(inputs, config=config)
    response = extract_response(result)
    
    # Process token counting
    process_token_counting(observer, token_data, result, response, model_name="gpt-4")
```

Token Counting Workflow:
-----------------------
1. Before agent invocation: Collect query, enhanced_query, system_prompt, and history
2. After agent invocation: Extract context from result and get response
3. Process all data: Update token_data with context and response, then process and finalize

Configuration:
-------------
Token counting can be enabled/disabled via:
- YAML config: `token_counting.enabled: true` in chatbot config file
- Environment variable: `ENABLE_TOKEN_COUNTING=true`
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import ToolMessage
from src.shared.config.logging import logger
from src.shared.utils.token_counter import get_token_counter, TokenCount


@dataclass
class ChatEvent:
    """Event data passed to token counting observer."""
    event_type: str  # 'query', 'enhanced_query', 'system_prompt', 'history', 'context', 'response'
    data: Any  # The actual text/messages/data
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenCountingObserver:
    """
    Observer that counts tokens for different components of a chat interaction.
    Receives events from the chatbot agent and accumulates token counts.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token counting observer.
        
        Args:
            model_name: Model name for accurate token counting
        """
        self.model_name = model_name or "default"
        self._token_counter = get_token_counter(self.model_name)
        self._components: List[TokenCount] = []
        self._events: List[ChatEvent] = []
        
    def on_event(self, event: ChatEvent) -> None:
        """
        Handle an event from the chatbot agent.
        
        Args:
            event: ChatEvent with event type and data
        """
        self._events.append(event)
        
        if event.event_type == 'query':
            count = self._token_counter.count_tokens(event.data, "User Query")
            self._components.append(count)
        elif event.event_type == 'enhanced_query':
            count = self._token_counter.count_tokens(event.data, "Enhanced Query (with topic)")
            self._components.append(count)
            # Also log original if available
            if 'original_query' in event.metadata:
                original_count = self._token_counter.count_tokens(
                    event.metadata['original_query'], 
                    "Original Query (before enhancement)"
                )
                logger.debug(
                    f"Query enhanced: {original_count.tokens} -> {count.tokens} tokens "
                    f"(+{count.tokens - original_count.tokens})"
                )
        elif event.event_type == 'system_prompt':
            count = self._token_counter.count_tokens(event.data, "System Prompt")
            self._components.append(count)
        elif event.event_type == 'history':
            count = self._token_counter.count_messages_tokens(event.data, "Conversation History")
            self._components.append(count)
        elif event.event_type == 'context':
            count = self._token_counter.count_tokens(event.data, "Retrieved Context")
            self._components.append(count)
        elif event.event_type == 'response':
            count = self._token_counter.count_tokens(event.data, "Response")
            self._components.append(count)
    
    def finalize(self, model_name: Optional[str] = None) -> None:
        """
        Finalize token counting and log the breakdown.
        Should be called after all events are received.
        
        Args:
            model_name: Model name for cost estimation (defaults to self.model_name)
        """
        # Separate response from input components
        response_count = None
        input_components = []
        
        for comp in self._components:
            if comp.component == "Response":
                response_count = comp
            else:
                input_components.append(comp)
        
        total_input_tokens = sum(comp.tokens for comp in input_components)
        total_output_tokens = response_count.tokens if response_count else 0
        
        # Log detailed token breakdown
        self._token_counter.log_token_breakdown(
            components=input_components,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            model_name=model_name or self.model_name
        )
    
    def reset(self) -> None:
        """Reset the observer for a new chat interaction."""
        self._components.clear()
        self._events.clear()
    
    def process_all_data(
        self,
        query: Optional[str] = None,
        enhanced_query: Optional[str] = None,
        original_query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Any]] = None,
        context: Optional[str] = None,
        response: Optional[str] = None
    ) -> None:
        """
        Process all token counting data in a single call.
        This is a cleaner alternative to multiple on_event() calls.
        
        Args:
            query: Original user query (if not enhanced)
            enhanced_query: Enhanced query with topic guidance
            original_query: Original query before enhancement (for comparison)
            system_prompt: System prompt text
            history: Conversation history messages
            context: Retrieved context from RAG
            response: Agent response
        """
        # Reset first
        self.reset()
        
        # Process query (use enhanced_query if available, otherwise original query)
        if enhanced_query:
            count = self._token_counter.count_tokens(enhanced_query, "Enhanced Query (with topic)")
            self._components.append(count)
            # Also log original if available and different
            if original_query and original_query != enhanced_query:
                original_count = self._token_counter.count_tokens(
                    original_query, 
                    "Original Query (before enhancement)"
                )
                self._components.append(original_count)
                logger.debug(
                    f"Query enhanced: {original_count.tokens} -> {count.tokens} tokens "
                    f"(+{count.tokens - original_count.tokens})"
                )
        elif query:
            count = self._token_counter.count_tokens(query, "User Query")
            self._components.append(count)
        
        # Process system prompt
        if system_prompt:
            count = self._token_counter.count_tokens(system_prompt, "System Prompt")
            self._components.append(count)
        
        # Process history
        if history:
            count = self._token_counter.count_messages_tokens(history, "Conversation History")
            self._components.append(count)
        
        # Process context
        if context:
            count = self._token_counter.count_tokens(context, "Retrieved Context")
            self._components.append(count)
        
        # Process response
        if response:
            count = self._token_counter.count_tokens(response, "Response")
            self._components.append(count)


class TokenCountingWrapper:
    """
    Wrapper that provides token counting functionality via observer pattern.
    Can be enabled/disabled via configuration.
    """
    
    def __init__(self, enabled: bool = False, model_name: Optional[str] = None):
        """
        Initialize token counting wrapper.
        
        Args:
            enabled: Whether token counting is enabled
            model_name: Model name for accurate token counting
        """
        self.enabled = enabled
        self.model_name = model_name or "default"
        self._observer = None
        
        if self.enabled:
            self._observer = TokenCountingObserver(model_name=self.model_name)
            logger.info(f"Token counting enabled for model: {self.model_name}")
    
    def get_observer(self) -> Optional[TokenCountingObserver]:
        """
        Get the token counting observer if enabled.
        
        Returns:
            TokenCountingObserver instance if enabled, None otherwise
        """
        return self._observer if self.enabled else None
    
    @classmethod
    def create_from_config(
        cls,
        config_manager: Optional[Any] = None,
        model_name: Optional[str] = None,
        default_enabled: bool = False
    ) -> 'TokenCountingWrapper':
        """
        Create TokenCountingWrapper from configuration.
        
        Args:
            config_manager: ChatbotConfigManager instance (optional)
            model_name: Model name for token counting
            default_enabled: Default value if config not found
        
        Returns:
            TokenCountingWrapper instance
        """
        enabled = default_enabled
        
        if config_manager:
            try:
                # Check for token_counting.enabled in config
                enabled = config_manager.get("token_counting.enabled", default_enabled)
                if isinstance(enabled, str):
                    enabled = enabled.lower() in ('true', '1', 'yes', 'on')
            except Exception as e:
                logger.debug(f"Could not read token_counting.enabled from config: {e}")
        
        # Also check environment variable
        import os
        env_enabled = os.getenv("ENABLE_TOKEN_COUNTING", "").lower()
        if env_enabled:
            enabled = env_enabled in ('true', '1', 'yes', 'on')
        
        return cls(enabled=enabled, model_name=model_name)


def should_enable_token_counting(config_manager: Optional[Any] = None) -> bool:
    """
    Check if token counting should be enabled based on configuration.
    
    Args:
        config_manager: ChatbotConfigManager instance (optional)
    
    Returns:
        True if token counting should be enabled, False otherwise
    """
    # Check config file
    if config_manager:
        try:
            enabled = config_manager.get("token_counting.enabled", False)
            if isinstance(enabled, str):
                enabled = enabled.lower() in ('true', '1', 'yes', 'on')
            if enabled:
                return True
        except Exception:
            pass
    
    # Check environment variable
    import os
    env_enabled = os.getenv("ENABLE_TOKEN_COUNTING", "").lower()
    if env_enabled:
        return env_enabled in ('true', '1', 'yes', 'on')
    
    return False


def extract_context_from_result(result: Any) -> Optional[str]:
    """
    Extract retrieved context from agent result for token counting.
    
    This function searches through the agent's result messages to find tool messages
    that contain retrieved context from RAG operations. It handles multiple formats:
    - LangChain ToolMessage objects
    - Dictionary representations of tool messages
    - Artifact-based context (from LangGraph)
    - Direct content strings
    
    Args:
        result: Agent invocation result, typically a dict with "messages" key containing
                a list of message objects (HumanMessage, AIMessage, ToolMessage, etc.)
    
    Returns:
        Context text if found (concatenated if multiple artifacts), None otherwise.
        Returns None if no tool messages are found or if context extraction fails.
    
    Example:
        >>> result = {
        ...     "messages": [
        ...         {"type": "tool", "content": "Retrieved context about leave policy..."}
        ...     ]
        ... }
        >>> context = extract_context_from_result(result)
        >>> print(context)
        "Retrieved context about leave policy..."
    """
    try:
        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                is_tool_message = (
                    isinstance(msg, ToolMessage) or
                    (isinstance(msg, dict) and msg.get("type") == "tool")
                )
                
                if not is_tool_message:
                    continue
                
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', '')
                
                if content:
                    response_metadata = None
                    if hasattr(msg, 'response_metadata'):
                        response_metadata = msg.response_metadata
                    elif isinstance(msg, dict):
                        response_metadata = msg.get('response_metadata', {})
                    
                    if response_metadata and 'artifact' in response_metadata:
                        artifact = response_metadata['artifact']
                        if isinstance(artifact, list) and artifact:
                            artifact_content_parts = []
                            for item in artifact:
                                if isinstance(item, dict):
                                    artifact_content_parts.append(item.get('content', ''))
                            if artifact_content_parts:
                                return '\n\n'.join(artifact_content_parts)
                    elif isinstance(content, str) and len(content) > 100:
                        return content
    except Exception as e:
        logger.debug(f"Could not extract retrieved context for token counting: {e}")
    
    return None


def collect_token_data(
    query: str,
    enhanced_query: str,
    thread_id: str,
    user_id: Optional[str],
    system_prompt: Optional[str]
) -> dict:
    """
    Collect all token counting data before agent invocation.
    
    This function gathers all the input data that will be sent to the LLM, including:
    - User query (original and/or enhanced)
    - System prompt
    - Conversation history from checkpointer
    
    The returned dictionary is used later to count tokens for each component.
    
    Args:
        query: Original user query before any enhancements
        enhanced_query: Enhanced query with topic-specific guidance prepended (if topic
                       was detected). Should be the same as query if no enhancement occurred.
        thread_id: Thread/session identifier for retrieving conversation history
        user_id: Optional user identifier for retrieving user-specific history
        system_prompt: System prompt text that defines the agent's behavior
    
    Returns:
        Dictionary with token counting data structure:
        {
            'query': str or None,              # Original query (if not enhanced)
            'enhanced_query': str or None,      # Enhanced query (if enhancement occurred)
            'original_query': str or None,      # Original query (if enhancement occurred)
            'system_prompt': str or None,       # System prompt text
            'history': List[Message] or None,   # Conversation history messages
            'context': None,                    # Will be filled after agent invocation
            'response': None                    # Will be filled after agent invocation
        }
    
    Example:
        >>> token_data = collect_token_data(
        ...     query="What is the leave policy?",
        ...     enhanced_query="[Topic Context: Leave policies]\\n\\nUser Question: What is the leave policy?",
        ...     thread_id="thread-123",
        ...     user_id="user-456",
        ...     system_prompt="You are a helpful HR assistant..."
        ... )
        >>> print(token_data['enhanced_query'])
        "[Topic Context: Leave policies]\\n\\nUser Question: What is the leave policy?"
    """
    token_data = {
        'query': None,
        'enhanced_query': None,
        'original_query': None,
        'system_prompt': None,
        'history': None,
        'context': None,
        'response': None
    }
    
    # Get conversation history from checkpointer
    try:
        from src.infrastructure.storage.checkpointing.manager import get_checkpointer_manager
        checkpointer_manager = get_checkpointer_manager()
        config = checkpointer_manager.get_config(thread_id, user_id)
        checkpoint = checkpointer_manager.checkpointer.get(config)
        if checkpoint and "channel_values" in checkpoint:
            history_messages = checkpoint["channel_values"].get("messages", [])
            if history_messages:
                token_data['history'] = history_messages
    except Exception as e:
        logger.debug(f"Could not retrieve history for token counting: {e}")
    
    # Set query data
    if enhanced_query != query:
        token_data['enhanced_query'] = enhanced_query
        token_data['original_query'] = query
    else:
        token_data['query'] = query
    
    # Set system prompt
    if system_prompt:
        token_data['system_prompt'] = system_prompt
    
    return token_data


def update_token_data_with_result(
    token_data: dict,
    result: Any,
    response: str
) -> None:
    """
    Update token data dictionary with context and response from agent result.
    
    This function extracts the retrieved context from the agent result and adds both
    the context and response to the token_data dictionary. This should be called
    after the agent has been invoked and the response has been extracted.
    
    Args:
        token_data: Dictionary containing token counting data (from collect_token_data).
                   This dictionary will be modified in-place by adding 'context' and
                   'response' keys.
        result: Agent invocation result containing messages and metadata
        response: Extracted response text from the agent result
    
    Returns:
        None (modifies token_data in-place)
    
    Example:
        >>> token_data = {'query': '...', 'system_prompt': '...', ...}
        >>> result = agent.invoke(inputs, config=config)
        >>> response = extract_response(result)
        >>> update_token_data_with_result(token_data, result, response)
        >>> print(token_data['context'])  # Retrieved context from RAG
        >>> print(token_data['response'])  # Agent's response
    """
    # Extract context from result and update token data
    context = extract_context_from_result(result)
    if context:
        token_data['context'] = context
    
    # Add response to token data
    token_data['response'] = response


def process_token_counting(
    observer: TokenCountingObserver,
    token_data: dict,
    result: Any,
    response: str,
    model_name: Optional[str] = None
) -> None:
    """
    Process token counting for a chat interaction.
    
    This is the main function that orchestrates the complete token counting workflow:
    1. Updates token_data with context and response from the agent result
    2. Processes all collected token data through the observer
    3. Finalizes token counting and logs the breakdown
    
    This function should be called after the agent has been invoked and the response
    has been extracted. It will count tokens for all components and log a detailed
    breakdown including input/output token counts and estimated costs.
    
    Args:
        observer: Token counting observer instance (from TokenCountingWrapper.get_observer())
        token_data: Dictionary containing token counting data (from collect_token_data).
                   Will be updated with context and response.
        result: Agent invocation result containing messages and metadata
        response: Extracted response text from the agent result
        model_name: Model name for cost estimation during finalization.
                   Defaults to observer.model_name if not provided.
    
    Returns:
        None (logs token breakdown to logger)
    
    Raises:
        AttributeError: If observer is not properly initialized
    
    Example:
        >>> from src.shared.utils.token_counting_wrapper import (
        ...     TokenCountingWrapper,
        ...     collect_token_data,
        ...     process_token_counting
        ... )
        >>> 
        >>> wrapper = TokenCountingWrapper(enabled=True, model_name="gpt-4")
        >>> observer = wrapper.get_observer()
        >>> 
        >>> # Before agent invocation
        >>> token_data = collect_token_data(
        ...     query="...", enhanced_query="...", thread_id="...",
        ...     user_id="...", system_prompt="..."
        ... )
        >>> 
        >>> # After agent invocation
        >>> result = agent.invoke(inputs, config=config)
        >>> response = extract_response(result)
        >>> 
        >>> # Process token counting
        >>> process_token_counting(observer, token_data, result, response, "gpt-4")
        # Logs: Token breakdown with input/output counts and estimated costs
    """
    # Update token data with context and response from result
    update_token_data_with_result(token_data, result, response)
    
    # Process all token data in a single call
    observer.process_all_data(**token_data)
    observer.finalize(model_name=model_name or observer.model_name)
