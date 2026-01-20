# Token Counting Guide

This guide covers the token counting functionality in the RAG Chatbot application, including configuration, usage, and implementation details.

## Overview

Token counting provides detailed tracking and logging of token usage across different components of a chat interaction. This helps monitor costs, optimize prompts, and understand token consumption patterns.

**Key Features**:
- Automatic token counting for all chat components
- Detailed breakdown by component (query, system prompt, history, context, response)
- Cost estimation based on model pricing
- Optional feature (can be enabled/disabled)
- Zero performance impact when disabled

## Components

### TokenCountingObserver
Observes chat events and accumulates token counts for different components.

### TokenCountingWrapper
Wrapper that manages observer lifecycle and configuration. Can be enabled/disabled via configuration.

### Utility Functions
Helper functions for collecting and processing token data:
- `collect_token_data()`: Gathers input data before agent invocation
- `extract_context_from_result()`: Extracts retrieved context from agent results
- `update_token_data_with_result()`: Updates token data with context and response
- `process_token_counting()`: Processes and logs token counts

## Configuration

### YAML Configuration

Add token counting configuration to your chatbot config file:

```yaml
token_counting:
  enabled: true  # Enable/disable token counting
```

**Example**: `config/chatbot/hr_chatbot_config.yaml`

```yaml
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

token_counting:
  enabled: true

# ... rest of config
```

### Environment Variable

You can also enable token counting via environment variable:

```bash
ENABLE_TOKEN_COUNTING=true
```

**Priority**: Environment variable overrides YAML configuration.

## How It Works

### Workflow

1. **Before Agent Invocation**:
   - Collects query (original and/or enhanced)
   - Collects system prompt
   - Retrieves conversation history from checkpointer

2. **After Agent Invocation**:
   - Extracts retrieved context from RAG operations
   - Extracts agent response

3. **Token Processing**:
   - Counts tokens for all components
   - Logs detailed breakdown
   - Provides cost estimates

### Integration

Token counting is automatically integrated into the `ChatbotAgent.chat()` method. When enabled, it:

- Collects token data before agent invocation
- Processes token counts after agent invocation
- Logs detailed breakdown with input/output counts and cost estimates

**No code changes required** - it works automatically when enabled in configuration.

## Token Breakdown

The token counting system tracks the following components:

### Input Components
- **User Query**: Original user question
- **Enhanced Query**: Query with topic-specific guidance (if topic detected)
- **System Prompt**: System prompt that defines agent behavior
- **Conversation History**: Previous messages in the conversation
- **Retrieved Context**: Context retrieved from RAG operations

### Output Components
- **Response**: Agent's generated response

### Example Output

```
Token Breakdown:
├─ User Query: 15 tokens
├─ Enhanced Query (with topic): 45 tokens
│  └─ Original Query (before enhancement): 15 tokens
├─ System Prompt: 120 tokens
├─ Conversation History: 200 tokens
├─ Retrieved Context: 350 tokens
└─ Response: 180 tokens

Total Input Tokens: 730
Total Output Tokens: 180
Total Tokens: 910

Estimated Cost (gpt-4): $0.0273
```

## Usage Examples

### Basic Usage

Token counting is automatic when enabled. No code changes needed:

```python
from src.domain.chatbot.hr_chatbot import HRChatbot

# Token counting is enabled via config
chatbot = HRChatbot.get_from_pool()
response = chatbot.chat(
    query="What is the leave policy?",
    thread_id="thread-123"
)
# Token breakdown is automatically logged
```

### Manual Usage (Advanced)

If you need to use token counting utilities directly:

```python
from src.shared.utils.token_counting_wrapper import (
    TokenCountingWrapper,
    collect_token_data,
    process_token_counting
)

# Initialize wrapper
wrapper = TokenCountingWrapper(enabled=True, model_name="gpt-4")
observer = wrapper.get_observer()

# Before agent invocation
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
process_token_counting(observer, token_data, result, response, "gpt-4")
```

## Architecture

### Module Location

Token counting utilities are located in:
- `src/shared/utils/token_counting_wrapper.py`: Core token counting functionality
- `src/shared/utils/token_counter.py`: Token counting implementation

### Integration Points

1. **ChatbotAgent Initialization**:
   - `_initialize_token_counting()`: Initializes token counting wrapper if enabled

2. **Chat Method**:
   - Collects token data before agent invocation
   - Processes token counts after agent invocation

3. **Utility Functions**:
   - Modular functions for collecting and processing token data
   - Can be used independently if needed

## Performance

- **When Disabled**: Zero overhead - no token counting code executes
- **When Enabled**: Minimal overhead - token counting happens asynchronously
- **Impact**: Negligible impact on response times (< 1ms typically)

## Troubleshooting

### Token Counting Not Working

1. **Check Configuration**:
   - Verify `token_counting.enabled: true` in YAML config
   - Or check `ENABLE_TOKEN_COUNTING` environment variable

2. **Check Logs**:
   - Look for "Token counting enabled" message in logs
   - Check for token breakdown logs after chat interactions

3. **Verify Observer**:
   - Ensure `_token_counting_wrapper` is not None
   - Check that observer is retrieved successfully

### Missing Token Data

- **History Not Collected**: Check checkpointer configuration
- **Context Not Extracted**: Verify RAG tools are enabled
- **Response Not Counted**: Ensure agent invocation succeeded

### Incorrect Token Counts

- **Model Mismatch**: Ensure model_name matches actual model used
- **Encoding Issues**: Token counter uses tiktoken for accurate counting
- **Content Format**: Some content formats may not be counted correctly

## Best Practices

1. **Enable in Development**: Use token counting during development to optimize prompts
2. **Monitor in Production**: Enable selectively in production for cost monitoring
3. **Review Logs Regularly**: Check token breakdowns to identify optimization opportunities
4. **Model-Specific**: Token counting is model-specific - ensure correct model_name

## Related Documentation

- [Configuration Guide](CONFIGURATION.md) - Configuration details
- [Architecture Guide](ARCHITECTURE.md) - System architecture
- [API Usage](API_USAGE.md) - API documentation
