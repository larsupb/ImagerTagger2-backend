import sqlite3

from app.db.connection import _check_and_migrate
from app.db.schema import create_schema
from app.db.repository import VectorRepository


def _db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    create_schema(db)
    db.execute("INSERT INTO images (id, filename) VALUES (1, 'a.png'), (2, 'b.png')")
    return db


def test_caption_vectors_table_exists():
    db = _db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(caption_vectors)")]
    assert cols == ["image_id", "model", "dim", "embedding", "text_hash", "updated_at"]


def test_upsert_and_load_roundtrip():
    db = _db()
    blob = (b"\x00" * 16)
    VectorRepository.upsert(db, 1, "m1", 4, blob, "hash-a")
    db.commit()
    rows = VectorRepository.load_all(db)
    assert len(rows) == 1
    assert bytes(rows[0]["embedding"]) == blob
    VectorRepository.upsert(db, 1, "m1", 4, b"\x01" * 16, "hash-b")
    db.commit()
    assert VectorRepository.hashes(db) == {1: ("m1", "hash-b")}
    assert len(VectorRepository.load_all(db)) == 1


def test_orphans_deleted_and_revision_changes():
    db = _db()
    VectorRepository.upsert(db, 1, "m1", 4, b"\x00" * 16, "h1")
    VectorRepository.upsert(db, 2, "m1", 4, b"\x00" * 16, "h2")
    db.commit()
    before = VectorRepository.revision(db)
    removed = VectorRepository.delete_missing(db, {1})
    db.commit()
    assert removed == 1
    assert VectorRepository.hashes(db) == {1: ("m1", "h1")}
    assert VectorRepository.revision(db) != before


def test_image_delete_cascades():
    db = _db()
    VectorRepository.upsert(db, 1, "m1", 4, b"\x00" * 16, "h1")
    db.commit()
    db.execute("DELETE FROM images WHERE id = 1")
    assert VectorRepository.hashes(db) == {}


def test_v2_db_migration_creates_caption_vectors():
    db = _db()
    db.execute("DROP TABLE caption_vectors")
    db.execute("UPDATE schema_info SET value = '2' WHERE key = 'schema_version'")
    _check_and_migrate(db)
    cols = [r[1] for r in db.execute("PRAGMA table_info(caption_vectors)")]
    assert cols
    version = db.execute(
        "SELECT value FROM schema_info WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert version == "3"
