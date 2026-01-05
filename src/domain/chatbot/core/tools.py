"""
Tool Factory for Chatbot Agents
Handles creation of tools based on configuration.
Follows Single Responsibility Principle.
"""
from typing import Optional, List
from langchain_core.tools import BaseTool
from src.shared.config.logging import logger
from src.domain.retrieval.service import RetrievalService
from src.infrastructure.vectorstore import get_vector_store
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys


class ChatbotToolFactory:
    """
    Factory for creating chatbot tools based on configuration.
    Handles retrieval tool creation and merging with provided tools.
    """
    
    def __init__(self, config_manager: Optional[ChatbotConfigManager] = None):
        """
        Initialize tool factory.
        
        Args:
            config_manager: Config manager instance (optional)
        """
        self.config_manager = config_manager
    
    def create_tools(
        self,
        chatbot_type: str,
        provided_tools: Optional[List[BaseTool]] = None
    ) -> List[BaseTool]:
        """
        Create tools for the agent from configuration.
        
        This method:
        1. Adds retrieval tool if enabled in config (tools.enable_retrieval)
        2. Merges with provided_tools, avoiding duplicates
        
        Args:
            chatbot_type: Type of chatbot (used as fallback for vector_store.type)
            provided_tools: Tools provided during initialization (if any)
        
        Returns:
            List of tools to use
        """
        tools = []
        
        # Add retrieval tool if enabled in config
        if self._should_enable_retrieval():
            retrieval_tool = self._create_retrieval_tool(chatbot_type)
            if retrieval_tool:
                tools.append(retrieval_tool)
        
        # Merge with provided tools, avoiding duplicates
        if provided_tools:
            existing_names = {tool.name for tool in tools}
            tools.extend(
                tool for tool in provided_tools 
                if tool.name not in existing_names
            )
        
        return tools
    
    def _should_enable_retrieval(self) -> bool:
        """Check if retrieval tool should be enabled."""
        if not self.config_manager:
            return False
        return self.config_manager.get(ConfigKeys.TOOLS_ENABLE_RETRIEVAL, False)
    
    def _create_retrieval_tool(self, chatbot_type: str) -> Optional[BaseTool]:
        """
        Create retrieval tool based on configuration.
        
        Args:
            chatbot_type: Type of chatbot (used as fallback for vector_store.type)
        
        Returns:
            Retrieval tool or None if creation fails
        """
        try:
            # Get vector store type from config or use chatbot_type as fallback
            vector_store_type = None
            if self.config_manager:
                vector_store_type = self.config_manager.get(ConfigKeys.VECTOR_STORE_TYPE)
            
            if not vector_store_type:
                vector_store_type = chatbot_type
            
            # Pass config_manager to get_vector_store so it can read vector store config
            # from the same config file that the chatbot uses
            vector_store = get_vector_store(vector_store_type, config_manager=self.config_manager)
            retrieval_service = RetrievalService(vector_store)
            return retrieval_service.create_tool()
        except Exception as e:
            logger.error(f"Failed to create retrieval tool: {e}", exc_info=True)
            return None

