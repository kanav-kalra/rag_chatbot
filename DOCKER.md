# Docker Guide

This guide covers Docker setup and usage for the RAG Chatbot application.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

1. **Create `.env` file** (if not already created):
   ```bash
   cp .env-sample .env
   ```

2. **Edit `.env` and add your API keys**:
   ```bash
   OPENAI_API_KEY=your-key-here
   # Add other API keys as needed
   ```

3. **Build and start services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the applications**:
   - FastAPI: http://localhost:8000
   - FastAPI Docs: http://localhost:8000/docs
   - Streamlit: http://localhost:8501

## Docker Services

### Application Service (`app`)
- **Image**: Built from `Dockerfile`
- **Ports**: 
  - 8000 (FastAPI)
  - 8501 (Streamlit)
- **Volumes**:
  - `./data` → `/app/data` (persistent storage for logs and vector stores)
  - `./config` → `/app/config:ro` (read-only config files)
- **Environment**: Loads from `.env` file

### Redis Service (`redis`)
- **Image**: `redis/redis-stack-server:latest`
- **Port**: 6379 (exposed to host)
- **Volume**: `redis_data` (persistent Redis data)
- **Purpose**: Session checkpointing and state management (includes RediSearch support)

## Common Commands

### Start Services
```bash
# Start in foreground (see logs)
docker-compose up

# Start in background (detached)
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Stop Services
```bash
# Stop services (keeps volumes)
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 app
```

### Execute Commands in Container
```bash
# Run a command in the app container
docker-compose exec app python -m src.main --app fastapi

# Open a shell in the container
docker-compose exec app /bin/bash
```

### Rebuild After Code Changes
```bash
# Rebuild and restart
docker-compose up --build

# Force rebuild without cache
docker-compose build --no-cache
docker-compose up
```

## Environment Variables

The application loads environment variables from `.env` file. Key variables:

- **API Keys** (Required):
  - `OPENAI_API_KEY` - Required for embeddings
  - `ANTHROPIC_API_KEY` - Optional, for Claude models
  - `GOOGLE_API_KEY` - Optional, for Gemini models

- **Redis** (Auto-configured in Docker):
  - `REDIS_URL=redis://redis:6379` - Automatically set in docker-compose.yml
  - You don't need to set this in `.env` when using Docker

- **Server Configuration**:
  - `HOST=0.0.0.0` - Already set in docker-compose.yml
  - `PORT=8000` - Already set in docker-compose.yml
  - `STREAMLIT_PORT=8501` - Already set in docker-compose.yml

## Data Persistence

### Persistent Volumes

- **`./data`**: Mounted to `/app/data` in container
  - Contains logs (`data/logs/`)
  - Contains vector stores (`data/vectorstores/`)
  - Data persists across container restarts

- **`redis_data`**: Docker volume for Redis data
  - Persists Redis checkpoint data
  - Removed only with `docker-compose down -v`

### Creating Vector Stores

To create vector stores, you can either:

1. **Run ingestion script in container**:
   ```bash
   docker-compose exec app python scripts/ingestion/create_vectorstore.py --chatbot-type hr
   ```

2. **Run locally** (if you have Python environment):
   ```bash
   python scripts/ingestion/create_vectorstore.py --chatbot-type hr
   ```
   The vector store will be saved to `./data/vectorstores/` which is mounted in the container.

## Development with Docker

### Hot Reload (Development)

For development with code hot-reload, create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  app:
    environment:
      - DEBUG=True
    volumes:
      - ./src:/app/src  # Mount source for hot-reload
      - ./config:/app/config
      - ./data:/app/data
```

**Note**: Hot-reload works for Python code changes, but you may need to restart for some changes (e.g., new dependencies).

### Running Specific Services

Edit `docker-compose.override.yml` to override the command:

```yaml
services:
  app:
    command: ["python", "-m", "src.main", "--app", "fastapi"]
    # or
    command: ["python", "-m", "src.main", "--app", "streamlit"]
```

## Troubleshooting

### Container Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs app
   ```

2. **Verify `.env` file exists**:
   ```bash
   ls -la .env
   ```

3. **Check Redis is healthy**:
   ```bash
   docker-compose ps
   docker-compose logs redis
   ```

### Redis Connection Errors

- Ensure Redis service is running: `docker-compose ps`
- Check Redis logs: `docker-compose logs redis`
- Verify `REDIS_URL` in docker-compose.yml is `redis://redis:6379`

### Port Already in Use

If ports 8000, 8501, or 6379 are already in use:

1. **Stop conflicting services**, or
2. **Change ports in `docker-compose.yml`**:
   ```yaml
   ports:
     - "8001:8000"  # Change host port
     - "8502:8501"
   ```

### Permission Errors

If you encounter permission errors with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER ./data

# Or run container with your user ID
# Add to docker-compose.yml:
user: "${UID}:${GID}"
```

### Rebuilding from Scratch

```bash
# Stop and remove everything
docker-compose down -v

# Remove images
docker-compose rm -f

# Rebuild
docker-compose build --no-cache
docker-compose up
```

## Production Considerations

For production deployment:

1. **Use specific image tags** instead of `latest`
2. **Set proper environment variables** (don't use `.env` file in production)
3. **Use secrets management** (Docker secrets, Kubernetes secrets, etc.)
4. **Configure resource limits** in docker-compose.yml:
   ```yaml
   services:
     app:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
   ```
5. **Use reverse proxy** (nginx, traefik) for SSL/TLS
6. **Set up log rotation** for persistent logs
7. **Use healthchecks** (already configured)
8. **Configure restart policies** (already set to `unless-stopped`)

## Multi-Architecture Support

The Dockerfile uses `python:3.12-slim` which supports:
- linux/amd64
- linux/arm64

For ARM-based systems (Apple Silicon, Raspberry Pi), Docker will automatically use the correct architecture.
