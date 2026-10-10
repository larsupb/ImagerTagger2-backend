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
