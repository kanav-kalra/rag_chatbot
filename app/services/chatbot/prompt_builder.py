"""
Prompt Builder for Chatbot Agents
Handles system prompt creation from configuration and prompts files.
Follows Single Responsibility Principle.
"""
import yaml
from pathlib import Path
from typing import Optional
from app.core.logging import logger
from app.services.chatbot.config_manager import ChatbotConfigManager, ConfigKeys


class ChatbotPromptBuilder:
    """
    Builder for creating system prompts from configuration.
    Handles loading prompts from YAML files and combining templates.
    """
    
    def __init__(self, config_manager: Optional[ChatbotConfigManager] = None):
        """
        Initialize prompt builder.
        
        Args:
            config_manager: Config manager instance (optional)
        """
        self.config_manager = config_manager
        self._prompts_dir = Path(__file__).parent
    
    def build_system_prompt(self) -> Optional[str]:
        """
        Build system prompt from configuration.
        If template/instructions are null, tries to use default prompts based on prompts_file.
        
        Returns:
            System prompt string or None
        """
        if not self.config_manager or not self.config_manager.has_config():
            return None
        
        # Get templates from config
        system_prompt_template = self.config_manager.get(ConfigKeys.SYSTEM_PROMPT_TEMPLATE)
        agent_instructions_template = self.config_manager.get(
            ConfigKeys.SYSTEM_PROMPT_AGENT_INSTRUCTIONS
        )
        
        # If null, try to load from prompts file
        if system_prompt_template is None or agent_instructions_template is None:
            prompts_data = self._load_prompts_file()
            if prompts_data:
                if system_prompt_template is None:
                    system_prompt_template = prompts_data.get("system_prompt")
                if agent_instructions_template is None:
                    agent_instructions_template = prompts_data.get("agent_instructions")
        
        # If both are still None, return None (no system prompt)
        if system_prompt_template is None and agent_instructions_template is None:
            return None
        
        # Combine template and instructions
        parts = []
        if system_prompt_template:
            parts.append(system_prompt_template)
        if agent_instructions_template:
            parts.append(agent_instructions_template)
        
        return "\n\n".join(parts) if parts else None
    
    def _load_prompts_file(self) -> Optional[dict]:
        """
        Load prompts from YAML file specified in config.
        
        Returns:
            Dictionary with prompts data or None if loading fails
        """
        if not self.config_manager:
            return None
        
        # Get prompts filename from config
        prompts_filename = self.config_manager.get(ConfigKeys.SYSTEM_PROMPT_PROMPTS_FILE)
        if not prompts_filename:
            # Try default prompts file
            prompts_filename = "default_prompts.yaml"
        
        prompts_file = self._prompts_dir / prompts_filename
        if not prompts_file.exists():
            logger.debug(f"Prompts file not found: {prompts_file}")
            return None
        
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompts_data = yaml.safe_load(f)
                return prompts_data if isinstance(prompts_data, dict) else None
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML in prompts file {prompts_file}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load prompts file {prompts_file}: {e}")
            return None

