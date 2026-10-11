import app.config as config


def test_embedding_endpoint_defaults_present():
    assert "embedding_endpoint" in config.DEFAULTS
    d = config.DEFAULTS["embedding_endpoint"]
    assert set(d) >= {"api_key", "base_url", "model", "timeout"}


def test_embedding_endpoint_settings_reads_settings_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    config.update_setting("embedding_endpoint", {"api_key": "k", "base_url": "http://x/v1",
                                                 "model": "embed-m", "timeout": 30})
    s = config.embedding_endpoint_settings()
    assert s["model"] == "embed-m" and s["base_url"] == "http://x/v1"


import numpy as np
import pytest

from app.services import embedding_service


SETTINGS = {"api_key": "k", "base_url": "http://x/v1", "model": "m", "timeout": 5}


def test_embed_texts_batches_and_normalizes(monkeypatch):
    calls = []

    def fake(texts, settings):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(embedding_service, "_create_embeddings", fake)
    out = embedding_service.embed_texts(["a", "b", "c"], SETTINGS)
    assert out.shape == (3, 2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_texts_retries_empty_response_once(monkeypatch):
    responses = [[], [[1.0, 1.0]]]
    monkeypatch.setattr(embedding_service, "_create_embeddings",
                        lambda texts, settings: responses.pop(0))
    out = embedding_service.embed_texts(["a"], SETTINGS)
    assert np.allclose(out[0], [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_embed_texts_raises_after_second_empty(monkeypatch):
    monkeypatch.setattr(embedding_service, "_create_embeddings",
                        lambda texts, settings: [])
    with pytest.raises(RuntimeError):
        embedding_service.embed_texts(["a"], SETTINGS)


def test_embed_texts_chunks_at_batch_size(monkeypatch):
    calls = []

    def fake(texts, settings):
        calls.append(len(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr(embedding_service, "_create_embeddings", fake)
    embedding_service.embed_texts([f"t{i}" for i in range(70)], SETTINGS)
    assert calls == [32, 32, 6]
