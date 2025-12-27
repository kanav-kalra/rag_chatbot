"""
API v1 Router
Combines all v1 route modules
"""
from fastapi import APIRouter

from app.api.v1.routes import items, users

api_router = APIRouter()

# Include all route modules
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(users.router, prefix="/users", tags=["users"])

