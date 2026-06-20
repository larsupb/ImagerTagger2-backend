import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.sessions import Session, get_session
from app.db.repository import ImageRepository, CategoryRepository

router = APIRouter(prefix="/api/categories", tags=["categories"])


class CategorySetRequest(BaseModel):
    category: Optional[str] = None


class CategoryRenameRequest(BaseModel):
    old_name: str
    new_name: str


class BulkCategoryRequest(BaseModel):
    indices: list[int]
    category: Optional[str] = None


@router.get("/")
def list_categories(session: Session = Depends(get_session)):
    if not session.db:
        return []
    return CategoryRepository.get_all(session.db)


@router.get("/item/{index}")
def get_category(index: int, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    item = ds.get_item(index)
    if not item or not session.db:
        return {"category": None}
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    img_row = ImageRepository.get_by_filename(session.db, rel_path)
    if not img_row:
        return {"category": None}
    return {"category": CategoryRepository.get_for_image(session.db, img_row["id"])}


@router.put("/item/{index}")
def set_category(index: int, req: CategorySetRequest, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    item = ds.get_item(index)
    if not item or not session.db:
        raise HTTPException(404, "Item not found")
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    img_id = ImageRepository.ensure_image(session.db, rel_path)
    category = req.category.strip() if req.category else None
    CategoryRepository.set_for_image(session.db, img_id, category or None)
    return {"category": category or None}


@router.post("/rename")
def rename_category(req: CategoryRenameRequest, session: Session = Depends(get_session)):
    if not session.db:
        raise HTTPException(400, "No dataset loaded")
    if not req.new_name.strip():
        raise HTTPException(400, "Category name cannot be empty")
    CategoryRepository.rename(session.db, req.old_name, req.new_name.strip())
    return {"ok": True}


@router.delete("/{name}")
def delete_category(name: str, session: Session = Depends(get_session)):
    if not session.db:
        raise HTTPException(400, "No dataset loaded")
    CategoryRepository.delete(session.db, name)
    return {"ok": True}


@router.post("/bulk")
def set_bulk_category(req: BulkCategoryRequest, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if not session.db:
        raise HTTPException(400, "No database")
    category = req.category.strip() if req.category else None
    updated = 0
    for index in req.indices:
        item = ds.get_item(index)
        if not item:
            continue
        rel_path = os.path.relpath(item.media_path, ds.base_dir)
        img_id = ImageRepository.ensure_image(session.db, rel_path)
        CategoryRepository.set_for_image(session.db, img_id, category or None)
        updated += 1
    return {"updated": updated, "category": category}
