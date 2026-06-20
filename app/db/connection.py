import os
import sqlite3

from app.db.schema import create_schema, SCHEMA_VERSION

PROJECT_CONFIG_DIR = ".imagetagger"
DB_FILENAME = "metadata.db"


def open_db(dataset_path: str) -> sqlite3.Connection:
    db_dir = os.path.join(dataset_path, PROJECT_CONFIG_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, DB_FILENAME)

    is_new = not os.path.exists(db_path)
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    if is_new:
        create_schema(db)
    else:
        _check_and_migrate(db)

    return db


def close_db(db: sqlite3.Connection) -> None:
    if db:
        db.close()


def _check_and_migrate(db) -> None:
    try:
        row = db.execute(
            "SELECT value FROM schema_info WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            create_schema(db)
            return
        version = int(row["value"])
    except sqlite3.OperationalError:
        create_schema(db)
        return

    if version < SCHEMA_VERSION:
        _run_migrations(db, version)


def _migrate_v1_to_v2(db) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    try:
        db.execute("ALTER TABLE images ADD COLUMN category_id INTEGER")
        db.commit()
    except Exception:
        pass


def _run_migrations(db, from_version: int) -> None:
    migrations = {
        1: _migrate_v1_to_v2,
    }
    for v in range(from_version, SCHEMA_VERSION):
        if v in migrations:
            migrations[v](db)
    db.execute(
        "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION),),
    )
    db.commit()
