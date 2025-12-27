"""
Users API Routes
"""
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter()

# Try to import EmailStr, fallback to str if email-validator not installed
try:
    from pydantic import EmailStr
except ImportError:
    EmailStr = str  # Fallback if email-validator not installed


# Pydantic models
class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True


class UserCreate(BaseModel):
    name: str
    email: str  # Using str instead of EmailStr for simplicity
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation"""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v


# In-memory storage (replace with database in production)
users_db = [
    {"id": 1, "name": "John Doe", "email": "john@example.com", "is_active": True},
    {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "is_active": True},
]


@router.get("/", response_model=List[User])
async def get_users():
    """Get all users"""
    return users_db


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: int):
    """Get a specific user by ID"""
    user = next((user for user in users_db if user["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/", response_model=User, status_code=201)
async def create_user(user: UserCreate):
    """Create a new user"""
    # Check if email already exists
    if any(u["email"] == user.email for u in users_db):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_id = max([user["id"] for user in users_db], default=0) + 1
    new_user = {"id": new_id, **user.dict(), "is_active": True}
    users_db.append(new_user)
    return new_user

