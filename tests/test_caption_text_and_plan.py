import hashlib
import sqlite3

from app.db.schema import create_schema
from app.db.repository import VectorRepository
from app.services.embedding_service import caption_text, compute_index_plan


def _db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    create_schema(db)
    db.execute("INSERT INTO images (id, filename) VALUES (1, 'a.png'), (2, 'b.png'), (3, 'c.png')")
    return db


def _caption(db, image_id, ctype, content):
    db.execute(
        "INSERT INTO captions (image_id, caption_type, content, is_active)"
        " VALUES (?, ?, ?, 1)"
        " ON CONFLICT(image_id, caption_type) DO UPDATE SET content = excluded.content",
        (image_id, ctype, content))


def test_caption_text_combines_all_types_sorted_ignores_empty():
    db = _db()
    _caption(db, 1, "tags", "1girl, waving")
    _caption(db, 1, "qwen3.8", "a person waving")
    _caption(db, 1, "empty_type", "")
    assert caption_text(db, 1) == "a person waving\n\n1girl, waving"


def test_plan_detects_new_changed_and_model_change():
    db = _db()
    _caption(db, 1, "tags", "hello")
    _caption(db, 2, "tags", "world")
    digest2 = hashlib.sha256(b"world").hexdigest()
    VectorRepository.upsert(db, 2, "m1", 2, b"\x00\x00", digest2)
    db.commit()

    todo, fresh = compute_index_plan(db, "m1")
    assert [i for i, _ in todo] == [1]
    assert fresh == 1

    _caption(db, 2, "tags", "world changed")
    todo, fresh = compute_index_plan(db, "m1")
    assert sorted(i for i, _ in todo) == [1, 2]

    todo, fresh = compute_index_plan(db, "other-model")
    assert len(todo) == 2
