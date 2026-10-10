import pytest

from app.db.connection import create_schema
from app.db.repository import AttributeRepository, ImageRepository
from app.attributes import ATTRIBUTES


@pytest.fixture()
def db(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    create_schema(conn)
    yield conn
    conn.close()


def _image(db, filename="a.png"):
    return ImageRepository.ensure_image(db, filename)


def test_set_get_unset_roundtrip(db):
    image_id = _image(db)
    AttributeRepository.set(db, image_id, "shot_type", "selfie")
    AttributeRepository.set(db, image_id, "pose", "standing")
    assert AttributeRepository.get_for_image(db, image_id) == {
        "shot_type": "selfie", "pose": "standing",
    }
    AttributeRepository.set(db, image_id, "shot_type", "headshot")
    assert AttributeRepository.get_for_image(db, image_id)["shot_type"] == "headshot"
    AttributeRepository.unset(db, image_id, "shot_type")
    assert AttributeRepository.get_for_image(db, image_id) == {"pose": "standing"}


def test_set_many_replaces_dict(db):
    image_id = _image(db)
    AttributeRepository.set_many(db, image_id, {"shot_type": "selfie", "pose": "standing"})
    AttributeRepository.set_many(db, image_id, {"pose": "sitting"})
    assert AttributeRepository.get_for_image(db, image_id) == {
        "shot_type": "selfie", "pose": "sitting",
    }


def test_attributes_cascade_on_image_delete(db):
    image_id = _image(db)
    AttributeRepository.set(db, image_id, "shot_type", "selfie")
    ImageRepository.delete_by_filename(db, "a.png")
    assert AttributeRepository.get_for_image(db, image_id) == {}


def test_get_all_rows_for_footprint(db):
    a, b = _image(db, "a.png"), _image(db, "b.png")
    AttributeRepository.set(db, a, "shot_type", "selfie")
    AttributeRepository.set(db, b, "shot_type", "selfie")
    rows = AttributeRepository.get_all_rows(db)
    assert rows == [("shot_type", "selfie", 2)]
