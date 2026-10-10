import sqlite3
from typing import Optional


class ImageRepository:
    @staticmethod
    def ensure_image(db: sqlite3.Connection, filename: str, **attrs) -> int:
        row = db.execute(
            "SELECT id FROM images WHERE filename = ?", (filename,)
        ).fetchone()
        if row:
            return row["id"]
        cols = ["filename"] + list(attrs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [filename] + list(attrs.values())
        cursor = db.execute(
            f"INSERT INTO images ({col_names}) VALUES ({placeholders})", values
        )
        db.commit()
        return cursor.lastrowid

    @staticmethod
    def get_by_filename(db: sqlite3.Connection, filename: str) -> Optional[dict]:
        row = db.execute(
            "SELECT * FROM images WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_by_id(db: sqlite3.Connection, image_id: int) -> Optional[dict]:
        row = db.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def update_metadata(
        db: sqlite3.Connection,
        image_id: int,
        width: int = None,
        height: int = None,
        file_size: int = None,
        file_hash: str = None,
    ) -> None:
        updates = []
        values = []
        if width is not None:
            updates.append("width = ?")
            values.append(width)
        if height is not None:
            updates.append("height = ?")
            values.append(height)
        if file_size is not None:
            updates.append("file_size = ?")
            values.append(file_size)
        if file_hash is not None:
            updates.append("file_hash = ?")
            values.append(file_hash)
        if not updates:
            return
        updates.append("updated_at = datetime('now')")
        values.append(image_id)
        db.execute(
            f"UPDATE images SET {', '.join(updates)} WHERE id = ?", values
        )
        db.commit()

    @staticmethod
    def set_bookmarked(db: sqlite3.Connection, image_id: int, value: bool) -> None:
        db.execute(
            "UPDATE images SET is_bookmarked = ?, updated_at = datetime('now') WHERE id = ?",
            (1 if value else 0, image_id),
        )
        db.commit()

    @staticmethod
    def is_bookmarked(db: sqlite3.Connection, filename: str) -> bool:
        row = db.execute(
            "SELECT is_bookmarked FROM images WHERE filename = ?", (filename,)
        ).fetchone()
        return bool(row["is_bookmarked"]) if row else False

    @staticmethod
    def get_all_filenames(db: sqlite3.Connection) -> set[str]:
        rows = db.execute("SELECT filename FROM images").fetchall()
        return {row["filename"] for row in rows}

    @staticmethod
    def delete_by_filename(db: sqlite3.Connection, filename: str) -> None:
        db.execute("DELETE FROM images WHERE filename = ?", (filename,))
        db.commit()

    @staticmethod
    def rename(db: sqlite3.Connection, old_filename: str, new_filename: str) -> None:
        db.execute(
            "UPDATE images SET filename = ?, updated_at = datetime('now') WHERE filename = ?",
            (new_filename, old_filename),
        )
        db.commit()


class CaptionRepository:
    @staticmethod
    def get(db: sqlite3.Connection, image_id: int, caption_type: str) -> Optional[str]:
        row = db.execute(
            "SELECT content FROM captions WHERE image_id = ? AND caption_type = ?",
            (image_id, caption_type),
        ).fetchone()
        return row["content"] if row else None

    @staticmethod
    def get_active(db: sqlite3.Connection, image_id: int) -> Optional[tuple]:
        row = db.execute(
            "SELECT caption_type, content FROM captions WHERE image_id = ? AND is_active = 1",
            (image_id,),
        ).fetchone()
        return (row["caption_type"], row["content"]) if row else None

    @staticmethod
    def get_all_for_image(db: sqlite3.Connection, image_id: int) -> list[dict]:
        rows = db.execute(
            "SELECT caption_type, content, is_active FROM captions WHERE image_id = ? ORDER BY caption_type",
            (image_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def upsert(
        db: sqlite3.Connection,
        image_id: int,
        caption_type: str,
        content: str,
        is_active: Optional[bool] = None,
    ) -> None:
        existing = db.execute(
            "SELECT id, is_active FROM captions WHERE image_id = ? AND caption_type = ?",
            (image_id, caption_type),
        ).fetchone()

        if existing:
            if is_active is not None:
                db.execute(
                    "UPDATE captions SET content = ?, is_active = ?, updated_at = datetime('now') WHERE id = ?",
                    (content, 1 if is_active else 0, existing["id"]),
                )
            else:
                db.execute(
                    "UPDATE captions SET content = ?, updated_at = datetime('now') WHERE id = ?",
                    (content, existing["id"]),
                )
        else:
            active = 1 if is_active else 0
            db.execute(
                "INSERT INTO captions (image_id, caption_type, content, is_active) VALUES (?, ?, ?, ?)",
                (image_id, caption_type, content, active),
            )
        db.commit()

    @staticmethod
    def set_active(db: sqlite3.Connection, image_id: int, caption_type: str) -> None:
        db.execute(
            "UPDATE captions SET is_active = 0 WHERE image_id = ?", (image_id,)
        )
        db.execute(
            "UPDATE captions SET is_active = 1 WHERE image_id = ? AND caption_type = ?",
            (image_id, caption_type),
        )
        db.commit()

    @staticmethod
    def get_all_by_type(db: sqlite3.Connection, caption_type: str) -> list[dict]:
        rows = db.execute(
            "SELECT i.filename, c.content FROM captions c JOIN images i ON c.image_id = i.id WHERE c.caption_type = ?",
            (caption_type,),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_all_types(db: sqlite3.Connection) -> list[str]:
        rows = db.execute(
            "SELECT DISTINCT caption_type FROM captions ORDER BY caption_type"
        ).fetchall()
        return [row["caption_type"] for row in rows]

    @staticmethod
    def rename_type(db: sqlite3.Connection, old_type: str, new_type: str) -> int:
        cursor = db.execute(
            "UPDATE captions SET caption_type = ? WHERE caption_type = ?",
            (new_type, old_type),
        )
        db.commit()
        return cursor.rowcount

    @staticmethod
    def delete_by_type(db: sqlite3.Connection, caption_type: str) -> int:
        cursor = db.execute(
            "DELETE FROM captions WHERE caption_type = ?", (caption_type,)
        )
        db.commit()
        return cursor.rowcount

    @staticmethod
    def get_all_active(db: sqlite3.Connection) -> list[dict]:
        rows = db.execute(
            "SELECT i.filename, c.caption_type, c.content FROM captions c JOIN images i ON c.image_id = i.id WHERE c.is_active = 1",
        ).fetchall()
        return [dict(r) for r in rows]


class VersionRepository:
    @staticmethod
    def create(
        db: sqlite3.Connection,
        image_id: int,
        version_path: str,
        operation: str,
        original_width: int = None,
        original_height: int = None,
        original_size: int = None,
    ) -> int:
        cursor = db.execute(
            "INSERT INTO image_versions (image_id, version_path, operation, original_width, original_height, original_size) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, version_path, operation, original_width, original_height, original_size),
        )
        db.commit()
        return cursor.lastrowid

    @staticmethod
    def get_for_image(db: sqlite3.Connection, image_id: int) -> list[dict]:
        rows = db.execute(
            "SELECT * FROM image_versions WHERE image_id = ? ORDER BY created_at DESC",
            (image_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_oldest_for_image(db: sqlite3.Connection, image_id: int) -> Optional[dict]:
        row = db.execute(
            "SELECT * FROM image_versions WHERE image_id = ? ORDER BY created_at ASC, id ASC LIMIT 1",
            (image_id,),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_by_id(db: sqlite3.Connection, version_id: int) -> Optional[dict]:
        row = db.execute(
            "SELECT * FROM image_versions WHERE id = ?", (version_id,)
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def delete(db: sqlite3.Connection, version_id: int) -> None:
        db.execute("DELETE FROM image_versions WHERE id = ?", (version_id,))
        db.commit()

    @staticmethod
    def delete_for_image(db: sqlite3.Connection, image_id: int) -> None:
        db.execute("DELETE FROM image_versions WHERE image_id = ?", (image_id,))
        db.commit()


class CategoryRepository:
    @staticmethod
    def get_all(db: sqlite3.Connection) -> list[str]:
        rows = db.execute("SELECT name FROM categories ORDER BY name").fetchall()
        return [row["name"] for row in rows]

    @staticmethod
    def get_or_create(db: sqlite3.Connection, name: str) -> int:
        row = db.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cursor = db.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        db.commit()
        return cursor.lastrowid

    @staticmethod
    def rename(db: sqlite3.Connection, old_name: str, new_name: str) -> None:
        db.execute("UPDATE categories SET name = ? WHERE name = ?", (new_name, old_name))
        db.commit()

    @staticmethod
    def delete(db: sqlite3.Connection, name: str) -> None:
        row = db.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        if row:
            db.execute("UPDATE images SET category_id = NULL WHERE category_id = ?", (row["id"],))
            db.execute("DELETE FROM categories WHERE id = ?", (row["id"],))
            db.commit()

    @staticmethod
    def get_for_image(db: sqlite3.Connection, image_id: int) -> Optional[str]:
        row = db.execute(
            "SELECT c.name FROM categories c JOIN images i ON i.category_id = c.id WHERE i.id = ?",
            (image_id,),
        ).fetchone()
        return row["name"] if row else None

    @staticmethod
    def get_filename_to_category(db: sqlite3.Connection) -> dict[str, str]:
        rows = db.execute(
            "SELECT i.filename, c.name FROM images i JOIN categories c ON i.category_id = c.id"
        ).fetchall()
        return {row["filename"]: row["name"] for row in rows}

    @staticmethod
    def set_for_image(db: sqlite3.Connection, image_id: int, category_name: Optional[str]) -> None:
        if category_name:
            cat_id = CategoryRepository.get_or_create(db, category_name)
            db.execute("UPDATE images SET category_id = ? WHERE id = ?", (cat_id, image_id))
        else:
            db.execute("UPDATE images SET category_id = NULL WHERE id = ?", (image_id,))
        db.commit()


class AttributeRepository:
    """Per-image categorical attributes stored in the metadata KV table."""

    @staticmethod
    def set(db: sqlite3.Connection, image_id: int, key: str, value: str) -> None:
        db.execute(
            "INSERT INTO metadata (image_id, key, value) VALUES (?, ?, ?)"
            " ON CONFLICT(image_id, key) DO UPDATE SET value = excluded.value",
            (image_id, key, value),
        )
        db.commit()

    @staticmethod
    def unset(db: sqlite3.Connection, image_id: int, key: str) -> None:
        db.execute(
            "DELETE FROM metadata WHERE image_id = ? AND key = ?", (image_id, key)
        )
        db.commit()

    @staticmethod
    def set_many(db: sqlite3.Connection, image_id: int, attributes: dict) -> None:
        for key, value in attributes.items():
            if value is None:
                db.execute(
                    "DELETE FROM metadata WHERE image_id = ? AND key = ?",
                    (image_id, key),
                )
            else:
                db.execute(
                    "INSERT INTO metadata (image_id, key, value) VALUES (?, ?, ?)"
                    " ON CONFLICT(image_id, key) DO UPDATE SET value = excluded.value",
                    (image_id, key, value),
                )
        db.commit()

    @staticmethod
    def get_for_image(db: sqlite3.Connection, image_id: int) -> dict:
        rows = db.execute(
            "SELECT key, value FROM metadata WHERE image_id = ?", (image_id,)
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    @staticmethod
    def get_for_image_ids(db: sqlite3.Connection, image_ids: list[int]) -> dict[int, dict]:
        if not image_ids:
            return {}
        placeholders = ", ".join("?" * len(image_ids))
        rows = db.execute(
            f"SELECT image_id, key, value FROM metadata WHERE image_id IN ({placeholders})",
            image_ids,
        ).fetchall()
        result: dict[int, dict] = {}
        for row in rows:
            result.setdefault(row["image_id"], {})[row["key"]] = row["value"]
        return result

    @staticmethod
    def get_all_rows(db: sqlite3.Connection) -> list[tuple]:
        rows = db.execute(
            "SELECT key, value, COUNT(*) AS n FROM metadata"
            " GROUP BY key, value ORDER BY key, value"
        ).fetchall()
        return [(row["key"], row["value"], row["n"]) for row in rows]

    @staticmethod
    def get_filenames_with_attributes(db: sqlite3.Connection) -> dict[str, dict]:
        rows = db.execute(
            "SELECT i.filename, m.key, m.value FROM metadata m JOIN images i ON m.image_id = i.id"
        ).fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            result.setdefault(row["filename"], {})[row["key"]] = row["value"]
        return result
