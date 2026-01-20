"""
Token counting utility for estimating token usage in LLM requests.
Supports multiple tokenization methods and provides cost estimation.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from src.shared.config.logging import logger


@dataclass
class TokenCount:
    """Token count information for a component."""
    component: str
    text: str
    tokens: int
    characters: int
    estimated_cost_usd: Optional[float] = None


class TokenCounter:
    """
    Token counter utility that estimates token counts for text.
    Uses tiktoken for accurate counting, falls back to character-based estimation.
    """
    
    # Common token-to-character ratios (approximate)
    # These are rough estimates and vary by model and text content
    TOKEN_RATIOS = {
        "gpt-4": 0.25,  # ~4 chars per token
        "gpt-3.5-turbo": 0.25,  # ~4 chars per token
        "gpt-4-turbo": 0.25,
        "gpt-4o": 0.25,
        "claude-3": 0.25,
        "claude-3-opus": 0.25,
        "claude-3-sonnet": 0.25,
        "claude-3-haiku": 0.25,
        "gemini": 0.25,
        "default": 0.25,  # ~4 chars per token (conservative estimate)
    }
    
    # Pricing per 1M tokens (as of 2024, approximate - update as needed)
    # Format: {model_name: {"input": price, "output": price}}
    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-pro": {"input": 0.5, "output": 1.5},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token counter.
        
        Args:
            model_name: Model name for tokenization (e.g., "gpt-4", "claude-3-opus")
                       If None, uses default estimation
        """
        self.model_name = model_name or "default"
        self._encoding = None
        
        # Try to get encoding for the model
        if TIKTOKEN_AVAILABLE and model_name:
            self._try_load_encoding(model_name)
    
    def _try_load_encoding(self, model_name: str) -> None:
        """Try to load tiktoken encoding for the model."""
        try:
            # Map model names to tiktoken encodings
            encoding_map = {
                "gpt-4": "cl100k_base",
                "gpt-4-turbo": "cl100k_base",
                "gpt-4o": "o200k_base",
                "gpt-3.5-turbo": "cl100k_base",
            }
            
            encoding_name = encoding_map.get(model_name.lower())
            if encoding_name:
                self._encoding = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Loaded tiktoken encoding '{encoding_name}' for model '{model_name}'")
        except Exception as e:
            logger.debug(f"Could not load tiktoken encoding for {model_name}: {e}. Using estimation.")
    
    def count_tokens(self, text: str, component_name: str = "text") -> TokenCount:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            component_name: Name of the component (for logging)
        
        Returns:
            TokenCount object with token and character counts
        """
        if not text:
            return TokenCount(
                component=component_name,
                text="",
                tokens=0,
                characters=0
            )
        
        characters = len(text)
        
        # Use tiktoken if available and encoding is loaded
        if self._encoding:
            try:
                tokens = len(self._encoding.encode(text))
            except Exception as e:
                logger.debug(f"Error encoding with tiktoken: {e}. Using estimation.")
                tokens = self._estimate_tokens(text)
        else:
            tokens = self._estimate_tokens(text)
        
        return TokenCount(
            component=component_name,
            text=text,
            tokens=tokens,
            characters=characters
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count based on character count.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count
        """
        ratio = self.TOKEN_RATIOS.get(self.model_name.lower(), self.TOKEN_RATIOS["default"])
        # tokens = characters / chars_per_token
        # chars_per_token = 1 / ratio
        chars_per_token = 1 / ratio
        return int(len(text) / chars_per_token)
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> float:
        """
        Estimate cost in USD for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model name (defaults to self.model_name)
        
        Returns:
            Estimated cost in USD
        """
        model = (model_name or self.model_name).lower()
        
        # Try to find matching pricing
        pricing = None
        for key, value in self.PRICING.items():
            if key.lower() in model or model in key.lower():
                pricing = value
                break
        
        if not pricing:
            logger.debug(f"No pricing found for model '{model}'. Cost estimation unavailable.")
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def count_messages_tokens(self, messages: List[Any], component_name: str = "messages") -> TokenCount:
        """
        Count tokens in a list of messages (LangChain message format).
        
        Args:
            messages: List of message objects (HumanMessage, AIMessage, etc.)
            component_name: Name of the component
        
        Returns:
            TokenCount object
        """
        if not messages:
            return TokenCount(
                component=component_name,
                text="",
                tokens=0,
                characters=0
            )
        
        total_text = ""
        for msg in messages:
            # Handle different message formats
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)
            
            if isinstance(content, str):
                total_text += content + "\n"
            elif isinstance(content, list):
                # Handle list content (e.g., Gemini format)
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        total_text += block['text'] + "\n"
                    else:
                        total_text += str(block) + "\n"
        
        return self.count_tokens(total_text, component_name)
    
    def log_token_breakdown(
        self,
        components: List[TokenCount],
        total_input_tokens: int,
        total_output_tokens: int,
        model_name: Optional[str] = None
    ) -> None:
        """
        Log a detailed token breakdown for cost optimization.
        
        Args:
            components: List of TokenCount objects for each component
            total_input_tokens: Total input tokens sent to LLM
            total_output_tokens: Total output tokens from LLM
            model_name: Model name for cost estimation
        """
        model = model_name or self.model_name
        total_cost = self.estimate_cost(total_input_tokens, total_output_tokens, model)
        
        logger.info("=" * 80)
        logger.info(f"TOKEN USAGE BREAKDOWN (Model: {model})")
        logger.info("=" * 80)
        
        for comp in components:
            percentage = (comp.tokens / total_input_tokens * 100) if total_input_tokens > 0 else 0
            logger.info(
                f"  {comp.component:30s}: {comp.tokens:6d} tokens "
                f"({comp.characters:6d} chars, {percentage:5.1f}% of input)"
            )
        
        logger.info("-" * 80)
        logger.info(f"  {'Total Input Tokens':30s}: {total_input_tokens:6d} tokens")
        logger.info(f"  {'Total Output Tokens':30s}: {total_output_tokens:6d} tokens")
        logger.info(f"  {'Total Tokens':30s}: {total_input_tokens + total_output_tokens:6d} tokens")
        logger.info(f"  {'Estimated Cost (USD)':30s}: ${total_cost:.6f}")
        logger.info("=" * 80)


def get_token_counter(model_name: Optional[str] = None) -> TokenCounter:
    """
    Get a TokenCounter instance for the specified model.
    
    Args:
        model_name: Model name (e.g., "gpt-4", "claude-3-opus")
    
    Returns:
        TokenCounter instance
    """
    return TokenCounter(model_name=model_name)
