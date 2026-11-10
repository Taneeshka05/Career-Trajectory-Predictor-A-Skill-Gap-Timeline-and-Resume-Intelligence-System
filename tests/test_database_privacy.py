import pytest

from core import database as db


def test_history_storage_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(db.HISTORY_ENV, raising=False)

    assert not db.history_enabled()


def test_disabled_history_does_not_create_database(tmp_path, monkeypatch):
    monkeypatch.delenv(db.HISTORY_ENV, raising=False)
    db_path = tmp_path / "careerpath.db"
    monkeypatch.setattr(db.cfg, "DB_PATH", str(db_path))

    db.init_db()

    assert not db_path.exists()


def test_history_reads_are_rejected_when_storage_is_disabled(monkeypatch):
    monkeypatch.delenv(db.HISTORY_ENV, raising=False)

    with pytest.raises(RuntimeError, match="History storage is disabled"):
        db.list_analyses("guest")


def test_history_storage_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv(db.HISTORY_ENV, "1")

    assert db.history_enabled()
