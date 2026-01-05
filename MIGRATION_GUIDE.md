# Migration Guide: Refactoring to New Folder Structure

This guide documents the refactoring from the old `app/` structure to the new `src/` structure following clean architecture principles.

## Overview

The codebase has been refactored to follow clean architecture with clear separation of concerns:
- **Domain Layer**: Core business logic (chatbot, memory, retrieval, session)
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External services (LLM, vector stores, storage)
- **API Layer**: FastAPI routes and middleware
- **UI Layer**: Streamlit application
- **Shared**: Common utilities and configuration

## Key Changes

### 1. Folder Structure

**Old Structure:**
```
app/
├── main.py
├── core/
├── services/
├── infra/
└── api/
```

**New Structure:**
```
src/
├── main.py
├── domain/
├── application/
├── infrastructure/
├── api/
├── ui/
└── shared/
```

### 2. Import Path Updates

All imports have been updated from `app.*` to `src.*`:

| Old Import | New Import |
|-----------|-----------|
| `from app.core.config import settings` | `from src.shared.config.settings import settings` |
| `from app.core.logging import logger` | `from src.shared.config.logging import logger` |
| `from app.services.chatbot.chatbot_service import ChatbotAgent` | `from src.domain.chatbot.core.chatbot_agent import ChatbotAgent` |
| `from app.infra.llm.llm_manager import get_llm_manager` | `from src.infrastructure.llm.manager import get_llm_manager` |

### 3. Path References

Path calculations have been updated to account for the new `src/` prefix:
- Old: `Path(__file__).parent.parent.parent` (3 levels)
- New: `Path(__file__).parent.parent.parent.parent.parent` (5 levels from deep files)

### 4. Configuration Files

Configuration YAML files have been moved:
- `app/core/hr_chatbot_config.yaml` → `config/chatbot/hr_chatbot_config.yaml`
- `app/services/chatbot/default_prompts.yaml` → `config/chatbot/prompts/default.yaml`
- `app/services/chatbot/hr_chatbot_prompts.yaml` → `config/chatbot/prompts/hr_chatbot.yaml`

### 5. Data Directories

- `chroma_db/` → `data/vectorstores/chroma_db/`
- `logs/` → `data/logs/`

### 6. Scripts

- `jobs/` → `scripts/ingestion/` and `scripts/jobs/`
- `ingestion/` → `src/application/ingestion/`

## Migration Steps

1. **Update Python Path**: Ensure `src/` is in your Python path
2. **Update Imports**: All imports have been updated to use `src.*`
3. **Update Config Paths**: Config file paths in code have been updated
4. **Update Data Paths**: Data directory paths have been updated
5. **Test**: Run tests to ensure everything works

## Running the Application

After migration, run the application using:

```bash
# From project root
python -m src.main --app both
```

Or update your IDE/launch configuration to use `src.main` instead of `app.main`.

## Backward Compatibility

**Note**: This refactoring does NOT maintain backward compatibility. All imports and paths must be updated.

## File Mapping Reference

See the main README.md for the complete file mapping table.

