import json
import os

from PIL import Image


class FakeItem:
    def __init__(self, path):
        self.media_path = path
        self.mask_path = None


class FakeDS:
    def __init__(self, img_dir, count):
        self.base_dir = str(img_dir.parent)
        self.is_initialized = True
        self._img_dir = img_dir
        self._count = count

    def __len__(self):
        return self._count

    def get_item(self, idx):
        return FakeItem(str(self._img_dir / f"img{idx:05d}.jpg"))


class FakeSession:
    def __init__(self, ds):
        self.dataset = ds
        self.db = None


def test_export_musubi_control_json_writes_json_files(tmp_path):
    from app.services.export_service import ExportOptions, export_dataset, MUSUBI_CONTROL_JSON_CONFIG

    ds_path = tmp_path / "dataset"
    ds_path.mkdir()
    img_dir = ds_path / "img"
    img_dir.mkdir()

    for i in range(2):
        Image.new("RGB", (512, 512), color="red").save(img_dir / f"img{i:05d}.jpg")

    ds = FakeDS(img_dir, 2)
    session = FakeSession(ds)

    from app.services import caption_service
    import pytest

    original_read = caption_service.read_caption

    def fake_read(sess, idx, ct):
        if idx == 0:
            return '{"high_level_description": "test 0"}'
        return '{"high_level_description": "test 1"}'

    caption_service.read_caption = fake_read
    try:
        options = ExportOptions(format="musubi_control_json", caption_type="tags")
        output_dir = export_dataset(session, options)

        captions_dir = os.path.join(output_dir, "captions")
        assert os.path.isdir(captions_dir), "captions dir should exist"
        assert os.path.isfile(os.path.join(captions_dir, "00000.json"))
        assert os.path.isfile(os.path.join(captions_dir, "00001.json"))

        with open(os.path.join(captions_dir, "00000.json")) as f:
            content = f.read()
        parsed = json.loads(content)
        assert parsed["high_level_description"] == "test 0"

        toml_path = os.path.join(output_dir, "dataset_config.toml")
        assert os.path.isfile(toml_path)
        with open(toml_path) as f:
            toml_content = f.read()
        assert 'caption_extension = ".json"' in toml_content
        assert 'caption_directory = "captions"' in toml_content
        assert toml_content == MUSUBI_CONTROL_JSON_CONFIG
    finally:
        caption_service.read_caption = original_read


def test_export_musubi_control_json_wraps_plain_text_captions(tmp_path):
    from app.services.export_service import ExportOptions, export_dataset

    ds_path = tmp_path / "dataset"
    ds_path.mkdir()
    img_dir = ds_path / "img"
    img_dir.mkdir()

    Image.new("RGB", (512, 512), color="blue").save(img_dir / "img00000.jpg")

    ds = FakeDS(img_dir, 1)
    session = FakeSession(ds)

    from app.services import caption_service

    original_read = caption_service.read_caption

    def fake_read(sess, idx, ct):
        return "plain text caption without json"

    caption_service.read_caption = fake_read
    try:
        options = ExportOptions(format="musubi_control_json", caption_type="tags")
        output_dir = export_dataset(session, options)

        captions_dir = os.path.join(output_dir, "captions")
        with open(os.path.join(captions_dir, "00000.json")) as f:
            parsed = json.load(f)
        assert parsed == {"caption": "plain text caption without json"}
    finally:
        caption_service.read_caption = original_read


def test_export_musubi_control_json_includes_control_masks(tmp_path):
    from app.services.export_service import ExportOptions, export_dataset

    ds_path = tmp_path / "dataset"
    ds_path.mkdir()
    img_dir = ds_path / "img"
    img_dir.mkdir()
    mask_dir = ds_path / "mask"
    mask_dir.mkdir()

    Image.new("RGB", (512, 512), color="red").save(img_dir / "img00000.jpg")
    Image.new("L", (512, 512), color=128).save(mask_dir / "img00000.png")

    class MaskItem:
        media_path = str(img_dir / "img00000.jpg")
        mask_path = str(mask_dir / "img00000.png")

    class MaskDS:
        base_dir = str(ds_path)
        is_initialized = True
        def __len__(self):
            return 1
        def get_item(self, idx):
            return MaskItem()

    session = FakeSession(MaskDS())

    from app.services import caption_service

    original_read = caption_service.read_caption
    caption_service.read_caption = lambda s, i, c: '{"high_level_description": "masked"}'
    try:
        options = ExportOptions(format="musubi_control_json", caption_type="tags")
        output_dir = export_dataset(session, options)

        control_dir = os.path.join(output_dir, "control")
        assert os.path.isdir(control_dir)
        assert os.path.isfile(os.path.join(control_dir, "00000.png"))
    finally:
        caption_service.read_caption = original_read