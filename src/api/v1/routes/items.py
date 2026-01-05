"""
Items API Routes
"""
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# Pydantic models for request/response
class Item(BaseModel):
    id: int
    name: str
    description: str = None
    price: float


class ItemCreate(BaseModel):
    name: str
    description: str = None
    price: float


# In-memory storage (replace with database in production)
items_db = [
    {"id": 1, "name": "Item 1", "description": "Description 1", "price": 10.99},
    {"id": 2, "name": "Item 2", "description": "Description 2", "price": 20.99},
]


@router.get("/", response_model=List[Item])
async def get_items():
    """Get all items"""
    return items_db


@router.get("/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get a specific item by ID"""
    item = next((item for item in items_db if item["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.post("/", response_model=Item, status_code=201)
async def create_item(item: ItemCreate):
    """Create a new item"""
    new_id = max([item["id"] for item in items_db], default=0) + 1
    new_item = {"id": new_id, **item.dict()}
    items_db.append(new_item)
    return new_item


@router.put("/{item_id}", response_model=Item)
async def update_item(item_id: int, item: ItemCreate):
    """Update an existing item"""
    item_index = next(
        (i for i, item in enumerate(items_db) if item["id"] == item_id), None
    )
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_index] = {"id": item_id, **item.dict()}
    return items_db[item_index]


@router.delete("/{item_id}", status_code=204)
async def delete_item(item_id: int):
    """Delete an item"""
    item_index = next(
        (i for i, item in enumerate(items_db) if item["id"] == item_id), None
    )
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db.pop(item_index)
    return None

