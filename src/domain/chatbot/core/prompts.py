"""
Prompt Builder for Chatbot Agents
Handles system prompt creation from configuration and prompts files.
Follows SOLID principles with separated concerns.
"""
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.shared.config.logging import logger
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys


class PromptFileLoader:
    """
    Responsible for loading and parsing prompt YAML files.
    Follows Single Responsibility Principle - only handles file I/O and parsing.
    """
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize file loader.
        
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self._prompts_dir = prompts_dir
    
    def get_file_path(self, filename: str) -> Path:
        """
        Get full path to a prompts file.
        
        Args:
            filename: Name of the prompts file
            
        Returns:
            Full path to the file
        """
        return self._prompts_dir / filename
    
    def load_prompts_file(self, filename: str) -> Dict[str, Any]:
        """
        Load and parse a prompts YAML file.
        
        Args:
            filename: Name of the prompts file
            
        Returns:
            Parsed YAML data as dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file doesn't contain valid dictionary
        """
        prompts_file = self.get_file_path(filename)
        if not prompts_file.exists():
            raise FileNotFoundError(
                f"Prompts file not found: {prompts_file}. "
                f"Please check the prompts_file setting in your config."
            )
        
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f)
            if not isinstance(prompts_data, dict):
                raise ValueError(f"Prompts file {prompts_file} does not contain a dictionary")
        
        return prompts_data
    
    @staticmethod
    def extract_nested_prompts(
        prompts_data: Dict[str, Any],
        required_keys: List[str],
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Extract prompts from nested structure (e.g., hr_chatbot.system_prompt).
        
        Args:
            prompts_data: Parsed YAML data
            required_keys: List of keys that should be present (e.g., ["system_prompt", "agent_instructions"])
            file_path: Optional file path for logging
            
        Returns:
            Extracted prompts data dictionary
        """
        # Check if required keys are already at top level
        if any(key in prompts_data for key in required_keys):
            return prompts_data
        
        # Search for nested structure
        for key, value in prompts_data.items():
            if isinstance(value, dict) and any(k in value for k in required_keys):
                if file_path:
                    logger.debug(f"Extracting prompts from nested key '{key}' in {file_path}")
                return value
        
        return prompts_data


class TopicDetector:
    """
    Responsible for detecting topics from user queries.
    Follows Single Responsibility Principle - only handles topic detection.
    Generic implementation that works with any topic keywords configuration.
    """
    
    def __init__(self, topic_keywords: Optional[Dict[str, List[str]]] = None):
        """
        Initialize topic detector.
        
        Args:
            topic_keywords: Optional mapping of topic names to keyword lists.
                          If None, topic detection will be disabled (returns None).
        """
        self._topic_keywords = topic_keywords or {}
    
    def detect(self, query: str, available_topics: Optional[List[str]] = None) -> Optional[str]:
        """
        Detect the most relevant topic from a user query using keyword matching.
        
        Args:
            query: User's query string
            available_topics: Optional list of available topics to filter by
            
        Returns:
            Topic identifier (e.g., "leave_policy", "benefits") or None if no topic detected
        """
        if not query or not self._topic_keywords:
            return None
        
        query_lower = query.lower()
        topic_scores = {}
        
        # Count keyword matches for each topic
        for topic, keywords in self._topic_keywords.items():
            if available_topics and topic not in available_topics:
                continue
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Return topic with highest score, or None if no matches
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Detected topic '{best_topic}' for query: '{query[:50]}...'")
            return best_topic
        
        return None


class PromptDataRepository:
    """
    Responsible for managing prompt data (topic prompts, RAG templates, topic keywords).
    Follows Single Responsibility Principle - only handles data storage and retrieval.
    """
    
    def __init__(self):
        """Initialize repository."""
        self._topic_prompts: Dict[str, str] = {}
        self._topic_keywords: Dict[str, List[str]] = {}
        self._rag_prompt_template: Optional[str] = None
    
    def set_topic_prompts(self, topic_prompts: Dict[str, str]) -> None:
        """Set topic prompts."""
        self._topic_prompts = topic_prompts or {}
    
    def get_topic_prompts(self) -> Dict[str, str]:
        """Get all topic prompts."""
        return self._topic_prompts
    
    def get_topic_prompt(self, topic: str) -> Optional[str]:
        """Get prompt for a specific topic."""
        return self._topic_prompts.get(topic)
    
    def get_available_topics(self) -> List[str]:
        """Get list of available topic identifiers."""
        return list(self._topic_prompts.keys())
    
    def set_topic_keywords(self, topic_keywords: Dict[str, List[str]]) -> None:
        """Set topic keywords for detection."""
        self._topic_keywords = topic_keywords or {}
    
    def get_topic_keywords(self) -> Dict[str, List[str]]:
        """Get all topic keywords."""
        return self._topic_keywords
    
    def set_rag_prompt_template(self, template: Optional[str]) -> None:
        """Set RAG prompt template."""
        self._rag_prompt_template = template
    
    def get_rag_prompt_template(self) -> Optional[str]:
        """Get RAG prompt template."""
        return self._rag_prompt_template


class ChatbotPromptBuilder:
    """
    Builder for creating system prompts from configuration.
    Coordinates between file loading, data storage, and prompt building.
    Follows Single Responsibility Principle - only handles prompt building orchestration.
    """
    
    def __init__(self, config_manager: Optional[ChatbotConfigManager] = None):
        """
        Initialize prompt builder.
        
        Args:
            config_manager: Config manager instance (optional)
        """
        self._config_manager = config_manager
        # Path from src/domain/chatbot/core/prompts.py -> config/chatbot/prompts/
        # Go up: core -> chatbot -> domain -> src -> project_root -> config/chatbot/prompts
        prompts_dir = Path(__file__).parent.parent.parent.parent.parent / "config" / "chatbot" / "prompts"
        
        # Initialize dependencies (Dependency Inversion Principle)
        self._file_loader = PromptFileLoader(prompts_dir)
        self._data_repository = PromptDataRepository()
        # TopicDetector will be initialized after loading keywords
        self._topic_detector: Optional[TopicDetector] = None
        
        # Load data during initialization
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Load topic prompts, topic keywords, and RAG template from configuration."""
        try:
            self._load_topic_prompts()
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load topic prompts: {e}")
            self._data_repository.set_topic_prompts({})
        
        try:
            self._load_topic_keywords()
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load topic keywords: {e}")
            self._data_repository.set_topic_keywords({})
        
        # Initialize topic detector with loaded keywords
        topic_keywords = self._data_repository.get_topic_keywords()
        self._topic_detector = TopicDetector(topic_keywords if topic_keywords else None)
        
        try:
            self._load_rag_prompt_template()
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load RAG prompt template: {e}")
            self._data_repository.set_rag_prompt_template(None)
    
    def _get_prompts_filename(self) -> Optional[str]:
        """Get prompts filename from config."""
        if not self._config_manager or not self._config_manager.has_config():
            return None
        return self._config_manager.get(ConfigKeys.SYSTEM_PROMPT_PROMPTS_FILE)
    
    def _load_prompts_data(self) -> Dict[str, Any]:
        """
        Load prompts data from file specified in config.
        
        Returns:
            Parsed prompts data
            
        Raises:
            ValueError: If config_manager is missing or prompts_file is not specified
            FileNotFoundError: If prompts file is not found
        """
        if not self._config_manager or not self._config_manager.has_config():
            raise ValueError("Config manager is required and must have a valid configuration")
        
        prompts_filename = self._get_prompts_filename()
        if not prompts_filename:
            raise ValueError(
                f"prompts_file is not specified in configuration. "
                f"Set {ConfigKeys.SYSTEM_PROMPT_PROMPTS_FILE} in your config."
            )
        
        return self._file_loader.load_prompts_file(prompts_filename)
    
    def _load_topic_prompts(self) -> None:
        """
        Load topic_prompts from the prompts YAML file.
        
        Raises:
            ValueError: If config_manager is missing or prompts_file is not specified
            FileNotFoundError: If the prompts file is not found
        """
        prompts_filename = self._get_prompts_filename()
        if not prompts_filename:
            self._data_repository.set_topic_prompts({})
            return
        
        try:
            prompts_data = self._file_loader.load_prompts_file(prompts_filename)
            prompts_file = self._file_loader.get_file_path(prompts_filename)
            
            # Extract from nested structure if needed
            prompts_data = self._file_loader.extract_nested_prompts(
                prompts_data,
                required_keys=["system_prompt", "agent_instructions", "topic_prompts"],
                file_path=prompts_file
            )
            
            topic_prompts = prompts_data.get("topic_prompts", {})
            self._data_repository.set_topic_prompts(topic_prompts)
            logger.debug(f"Loaded {len(topic_prompts)} topic prompts")
        except FileNotFoundError:
            self._data_repository.set_topic_prompts({})
            logger.debug("No topic_prompts found - prompts file not available")
    
    def _load_topic_keywords(self) -> None:
        """
        Load topic_keywords from the prompts YAML file.
        
        Raises:
            ValueError: If config_manager is missing or prompts_file is not specified
            FileNotFoundError: If the prompts file is not found
        """
        prompts_filename = self._get_prompts_filename()
        if not prompts_filename:
            self._data_repository.set_topic_keywords({})
            return
        
        try:
            prompts_data = self._file_loader.load_prompts_file(prompts_filename)
            prompts_file = self._file_loader.get_file_path(prompts_filename)
            
            # Extract from nested structure if needed
            prompts_data = self._file_loader.extract_nested_prompts(
                prompts_data,
                required_keys=["topic_keywords"],
                file_path=prompts_file
            )
            
            topic_keywords = prompts_data.get("topic_keywords", {})
            if topic_keywords and isinstance(topic_keywords, dict):
                self._data_repository.set_topic_keywords(topic_keywords)
                logger.debug(f"Loaded {len(topic_keywords)} topic keyword mappings")
            else:
                self._data_repository.set_topic_keywords({})
                logger.debug("No topic_keywords found in prompts file")
        except FileNotFoundError:
            self._data_repository.set_topic_keywords({})
            logger.debug("Topic keywords not found - prompts file not available")
    
    def _load_rag_prompt_template(self) -> None:
        """
        Load rag_prompt_template from the prompts YAML file.
        
        Raises:
            ValueError: If config_manager is missing or prompts_file is not specified
            FileNotFoundError: If the prompts file is not found
        """
        prompts_filename = self._get_prompts_filename()
        if not prompts_filename:
            self._data_repository.set_rag_prompt_template(None)
            return
        
        try:
            prompts_data = self._file_loader.load_prompts_file(prompts_filename)
            prompts_file = self._file_loader.get_file_path(prompts_filename)
            
            # Extract from nested structure if needed
            prompts_data = self._file_loader.extract_nested_prompts(
                prompts_data,
                required_keys=["rag_prompt_template"],
                file_path=prompts_file
            )
            
            rag_template = prompts_data.get("rag_prompt_template")
            self._data_repository.set_rag_prompt_template(rag_template)
            if rag_template:
                logger.debug("Loaded RAG prompt template from prompts file")
            else:
                logger.debug("No rag_prompt_template found in prompts file")
        except FileNotFoundError:
            self._data_repository.set_rag_prompt_template(None)
            logger.debug("RAG template not found - prompts file not available")
    
    def build_system_prompt(self, topic: Optional[str] = None) -> str:
        """
        Build system prompt from configuration, optionally including topic-specific guidance.
        
        Args:
            topic: Optional topic identifier (e.g., "leave_policy", "benefits") to include
                   topic-specific prompt guidance. If None, builds standard prompt.
        
        Returns:
            System prompt string
            
        Raises:
            ValueError: If config_manager is missing or required templates are not found
        """
        if not self._config_manager or not self._config_manager.has_config():
            raise ValueError("Config manager is required and must have a valid configuration")
        
        # Get templates from config
        system_prompt_template = self._config_manager.get(ConfigKeys.SYSTEM_PROMPT_TEMPLATE)
        agent_instructions_template = self._config_manager.get(
            ConfigKeys.SYSTEM_PROMPT_AGENT_INSTRUCTIONS
        )
        
        # Load from prompts file if templates are missing
        if system_prompt_template is None or agent_instructions_template is None:
            prompts_data = self._load_prompts_data()
            prompts_filename = self._get_prompts_filename()
            prompts_file = self._file_loader.get_file_path(prompts_filename) if prompts_filename else None
            
            # Extract from nested structure if needed
            prompts_data = self._file_loader.extract_nested_prompts(
                prompts_data,
                required_keys=["system_prompt", "agent_instructions"],
                file_path=prompts_file
            )
            
            # Load missing values from prompts file
            if system_prompt_template is None:
                system_prompt_template = prompts_data.get("system_prompt")
                if system_prompt_template is None:
                    raise ValueError(
                        "system_prompt_template is missing in configuration and "
                        "system_prompt is missing in prompts file. "
                        "At least one must be provided."
                    )
            
            if agent_instructions_template is None:
                agent_instructions_template = prompts_data.get("agent_instructions")
                if agent_instructions_template is None:
                    raise ValueError(
                        "agent_instructions_template is missing in configuration and "
                        "agent_instructions is missing in prompts file. "
                        "At least one must be provided."
                    )
            
            logger.debug("Loaded missing prompt templates from prompts file")
        
        # Validate that at least one is present
        if system_prompt_template is None and agent_instructions_template is None:
            raise ValueError(
                "Both system_prompt_template and agent_instructions_template are missing. "
                "At least one must be provided in the configuration or prompts file."
            )
        
        # Combine template and instructions
        parts = []
        if system_prompt_template:
            parts.append(system_prompt_template)
        
        # Add topic-specific prompt if topic is provided and exists
        if topic:
            topic_prompt = self._data_repository.get_topic_prompt(topic)
            if topic_prompt:
                parts.append(f"\n\nTopic-Specific Guidance:\n{topic_prompt}")
                logger.debug(f"Including topic-specific prompt for '{topic}'")
        
        if agent_instructions_template:
            parts.append(agent_instructions_template)
        
        return "\n\n".join(parts)
    
    def detect_topic(self, query: str) -> Optional[str]:
        """
        Detect the most relevant topic from a user query.
        
        Args:
            query: User's query string
        
        Returns:
            Topic identifier (e.g., "leave_policy", "benefits") or None if no topic detected
        """
        if not self._topic_detector:
            return None
        available_topics = self._data_repository.get_available_topics()
        return self._topic_detector.detect(query, available_topics)
    
    def get_available_topics(self) -> List[str]:
        """
        Get list of available topic identifiers.
        
        Returns:
            List of topic identifiers (e.g., ["leave_policy", "benefits", ...])
        """
        return self._data_repository.get_available_topics()
    
    def get_topic_prompt(self, topic: str) -> Optional[str]:
        """
        Get the prompt text for a specific topic.
        
        Args:
            topic: Topic identifier (e.g., "leave_policy", "benefits")
        
        Returns:
            Topic prompt text or None if topic doesn't exist
        """
        return self._data_repository.get_topic_prompt(topic)
    
    def get_rag_prompt_template(self) -> Optional[str]:
        """
        Get the RAG prompt template from the prompts file.
        
        Returns:
            RAG prompt template string with {context} and {question} placeholders, or None if not found
        """
        return self._data_repository.get_rag_prompt_template()
    
    def format_rag_prompt(self, context: str, question: str) -> Optional[str]:
        """
        Format the RAG prompt template with provided context and question.
        
        Args:
            context: Retrieved document context
            question: User's question
        
        Returns:
            Formatted RAG prompt string, or None if template is not available
        """
        template = self._data_repository.get_rag_prompt_template()
        if not template:
            return None
        
        try:
            return template.format(context=context, question=question)
        except KeyError as e:
            logger.warning(f"RAG prompt template missing required placeholder: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to format RAG prompt: {e}")
            return None
