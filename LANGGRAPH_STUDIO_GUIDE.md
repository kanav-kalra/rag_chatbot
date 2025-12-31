# LangGraph Studio Guide

This guide explains how to use LangGraph Studio for visualizing and debugging your chatbot agents.

## Prerequisites

1. **Install LangGraph CLI** (if not already installed):
   ```bash
   # Python >= 3.11 is required
   pip install --upgrade "langgraph-cli[inmem]"
   ```

2. **Ensure your `.env` file is configured** with:
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGSMITH_API_KEY=your-api-key` (preferred, per LangSmith Studio docs)
   - `LANGCHAIN_API_KEY=your-api-key` (also supported for backward compatibility)
   - `LANGCHAIN_PROJECT=rag-chatbot`
   - `LANGSMITH_WORKSPACE_ID=your-workspace-id` (only if using org-scoped API key)
   - Other required environment variables (OpenAI API key, Redis URL, etc.)

## Available Graphs

The `langgraph.json` file defines two graphs that you can visualize:

1. **`hr_chatbot`** - HR chatbot with RAG retrieval tool
2. **`default_chatbot`** - Basic chatbot without tools

## Using LangGraph Studio

### Method 1: Using LangGraph CLI

1. **Start LangGraph Studio**:
   ```bash
   langgraph dev
   ```
   
   **Note**: Safari blocks `localhost` connections to Studio. If using Safari, run with `--tunnel`:
   ```bash
   langgraph dev --tunnel
   ```

2. **Open Studio in your browser**:
   - The development server runs at `http://127.0.0.1:2024`
   - Access Studio UI at: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
   - Or check the terminal output for the exact URL

3. **Select a graph**:
   - Use the graph selector to choose between `hr_chatbot` and `default_chatbot`

4. **Visualize and debug**:
   - See the graph structure
   - Step through execution
   - Inspect intermediate states
   - View tool calls and LLM interactions

### Method 2: Using LangGraph Cloud

1. **Deploy your graph**:
   ```bash
   langgraph deploy
   ```

2. **Access via LangGraph Cloud dashboard**:
   - View graphs in the cloud
   - Monitor executions
   - Debug production issues

## Graph Factory Functions

The graph factory functions (`app/services/chatbot/graph_factory.py`) create agent instances that match your production setup:

- **`get_hr_chatbot_graph()`** - Creates HR chatbot with:
  - Retrieval tool for RAG
  - HR-specific system prompts
  - Checkpointer for memory

- **`get_default_chatbot_graph()`** - Creates basic chatbot with:
  - No tools
  - Checkpointer for memory

## Customizing Graphs in Studio

You can modify the graph factory functions to:
- Change model parameters
- Add/remove tools
- Modify prompts
- Adjust checkpointer settings

Changes in Studio are for visualization/debugging only and won't affect your production code.

## Troubleshooting

### Graph not loading
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that your `.env` file is in the project root
- Verify that Redis is running (if using Redis checkpointer)
- Ensure `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` is set in your `.env` file
- Check that `LANGCHAIN_TRACING_V2=true` is set if you want tracing enabled

### Import errors
- Make sure you're running `langgraph dev` from the project root directory
- Check that all Python paths are correctly configured
- Verify that the graph paths in `langgraph.json` are correct (e.g., `app.services.chatbot.graph_factory:hr_chatbot`)
- Ensure the graph variables (`hr_chatbot`, `default_chatbot`) are exported correctly from `graph_factory.py`

### Checkpointer issues
- If Redis is not available, the graph will fall back to in-memory checkpointer
- For production debugging, ensure Redis is accessible

## Integration with LangSmith

LangGraph Studio works seamlessly with LangSmith:
- All traces from Studio sessions appear in LangSmith
- You can view detailed execution traces
- Compare different graph runs
- Analyze performance and costs

## Next Steps

1. **Explore the graph structure** - Understand how your agent flows work
2. **Test different inputs** - See how the agent handles various queries
3. **Debug tool calls** - Inspect when and how tools are invoked
4. **Monitor in LangSmith** - View comprehensive traces and analytics

For more information, visit:
- [LangGraph Studio Documentation](https://docs.langchain.com/langgraph/studio)
- [LangGraph Platform Guide](https://docs.langchain.com/langgraph-platform)

