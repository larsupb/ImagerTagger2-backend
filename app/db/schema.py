SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    file_hash TEXT,
    is_bookmarked INTEGER NOT NULL DEFAULT 0,
    category_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS captions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    caption_type TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(image_id, caption_type)
);

CREATE INDEX IF NOT EXISTS idx_captions_type_active ON captions(caption_type, is_active);

CREATE TABLE IF NOT EXISTS image_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    version_path TEXT NOT NULL,
    operation TEXT NOT NULL,
    original_width INTEGER,
    original_height INTEGER,
    original_size INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_versions_image ON image_versions(image_id);

CREATE TABLE IF NOT EXISTS metadata (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY(image_id, key)
);
"""


def create_schema(db) -> None:
    db.executescript(SCHEMA_SQL)
    db.execute(
        "INSERT OR IGNORE INTO schema_info (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    db.commit()
