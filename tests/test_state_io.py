"""Tests for shared JSON state IO and capture dedup persistence."""

import json
from concurrent.futures import ThreadPoolExecutor

from capture.extract import load_processed, save_processed
from lib.common import read_json_file, update_json_file, write_json_file


class TestJsonStateIo:
    def test_read_json_returns_default_for_missing_file(self, tmp_path):
        path = tmp_path / "missing.json"
        result = read_json_file(path, {"processed": {}})
        assert result == {"processed": {}}

    def test_write_json_persists_newline_terminated_json(self, tmp_path):
        path = tmp_path / "state.json"
        write_json_file(path, {"a": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_update_json_is_thread_safe(self, tmp_path):
        path = tmp_path / "state.json"

        def _append(idx: int):
            def updater(data):
                data.append(idx)
            update_json_file(path, [], updater)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_append, range(20)))

        result = json.loads(path.read_text(encoding="utf-8"))
        assert sorted(result) == list(range(20))


class TestProcessedJson:
    def test_load_processed_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("capture.extract.PROCESSED_PATH", tmp_path / "processed.json")
        assert load_processed() == set()

    def test_save_then_load_processed_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("capture.extract.PROCESSED_PATH", tmp_path / "processed.json")
        save_processed({"a", "b"})
        assert load_processed() == {"a", "b"}

    def test_processed_json_is_valid_structure(self, tmp_path, monkeypatch):
        path = tmp_path / "processed.json"
        monkeypatch.setattr("capture.extract.PROCESSED_PATH", path)
        save_processed({"abc"})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"processed_ids": ["abc"]}
