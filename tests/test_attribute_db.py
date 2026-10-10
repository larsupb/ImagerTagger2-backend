import pytest

from app.db.connection import create_schema
from app.db.repository import AttributeRepository, CaptionRepository, ImageRepository
from app.attributes import ATTRIBUTES
from app.services.attribute_service import (
    UnknownAttributeError,
    compute_footprint,
    query_images,
    set_attributes,
)


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


def test_set_attributes_validates(db):
    image_id = _image(db)
    valid, errors = set_attributes(db, image_id, {"shot_type": "Selfie", "framing": "nope"})
    assert valid == {"shot_type": "selfie"}
    assert errors == {"framing": "nope"}
    assert AttributeRepository.get_for_image(db, image_id) == {"shot_type": "selfie"}


def test_footprint_counts_and_unset(db):
    a, b, c = _image(db, "a.png"), _image(db, "b.png"), _image(db, "c.png")
    AttributeRepository.set(db, a, "shot_type", "selfie")
    AttributeRepository.set(db, b, "shot_type", "selfie")
    AttributeRepository.set(db, c, "shot_type", "landscape")
    fp = compute_footprint(db)
    assert fp["_total"] == 3
    assert fp["shot_type"] == {"selfie": 2, "landscape": 1, "_unset": 0}
    assert fp["pose"]["_unset"] == 3
    assert set(fp) == {"_total", *ATTRIBUTES}


def test_query_and_or_unset(db):
    a = _image(db, "a.png")
    b = _image(db, "b.png")
    c = _image(db, "c.png")
    AttributeRepository.set_many(db, a, {"shot_type": "selfie", "pose": "walking"})
    AttributeRepository.set_many(db, b, {"shot_type": "selfie", "pose": "standing"})
    AttributeRepository.set(db, c, "shot_type", "selfie")
    CaptionRepository.upsert(db, a, "tags", "1girl, selfie")

    result = query_images(db, {"shot_type": ["selfie"], "pose": ["walking"]})
    assert [i["filename"] for i in result["items"]] == ["a.png"]
    assert result["items"][0]["captions"] == {"tags": "1girl, selfie"}
    assert result["total"] == 1

    result = query_images(db, {"shot_type": ["selfie"], "pose": ["walking", "standing"]})
    assert result["total"] == 2

    result = query_images(db, {"shot_type": ["selfie"], "pose": ["unset"]})
    assert [i["filename"] for i in result["items"]] == ["c.png"]
    assert result["total"] == 1

    result = query_images(db, {}, limit=1, offset=1)
    assert result["total"] == 3
    assert len(result["items"]) == 1


def test_query_unknown_attribute_raises(db):
    with pytest.raises(UnknownAttributeError):
        query_images(db, {"shot_typ": ["selfie"]})


def test_get_for_image_ids_batch(db):
    a, b = _image(db, "a.png"), _image(db, "b.png")
    AttributeRepository.set(db, a, "pose", "standing")
    AttributeRepository.set(db, b, "pose", "sitting")
    assert AttributeRepository.get_for_image_ids(db, [a, b, 999]) == {
        a: {"pose": "standing"}, b: {"pose": "sitting"},
    }
    assert AttributeRepository.get_for_image_ids(db, []) == {}
