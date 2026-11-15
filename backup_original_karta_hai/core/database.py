import os
import json
import sqlite3
from datetime import datetime
from contextlib import closing
from core import config as cfg
from core.logger_setup import get_logger

log = get_logger("database")

HISTORY_ENV = "CAREERPATH_ENABLE_HISTORY"


def history_enabled():
    return os.getenv(HISTORY_ENV, "").strip() == "1"


def _require_history():
    if not history_enabled():
        raise RuntimeError(
            f"History storage is disabled. Set {HISTORY_ENV}=1 only for a trusted, private deployment."
        )


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    profile TEXT NOT NULL,
    job_id INTEGER,
    job_title TEXT,
    domain TEXT,
    match_pct REAL,
    skill_match_pct REAL,
    level TEXT,
    weeks REAL,
    lang_score REAL,
    strength REAL,
    years INTEGER,
    resume_text TEXT,
    missing_json TEXT,
    plan_json TEXT
);
CREATE TABLE IF NOT EXISTS progress (
    analysis_id INTEGER NOT NULL,
    skill TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (analysis_id, skill)
);
"""


def connect():
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    if not history_enabled():
        return
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def save_analysis(profile, result, plan):
    _require_history()
    with closing(connect()) as conn:
        cur = conn.execute(
            "INSERT INTO analyses (created_at, profile, job_id, job_title, domain, match_pct, "
            "skill_match_pct, level, weeks, lang_score, strength, years, resume_text, "
            "missing_json, plan_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), profile, result["job_id"],
             result["job_title"], result["domain"], result["match_pct"],
             result["skill_match_pct"], result["level"], result["weeks"],
             result["lang_prob"] * 100, result["strength"], result["years"],
             result["resume_text"], json.dumps(result["missing"]), json.dumps(plan)))
        analysis_id = cur.lastrowid
        for item in plan:
            conn.execute("INSERT OR IGNORE INTO progress (analysis_id, skill, done) VALUES (?,?,0)",
                         (analysis_id, item["skill"]))
        conn.commit()
    log.info("Saved analysis %d for profile '%s'", analysis_id, profile)
    return analysis_id


def list_analyses(profile):
    _require_history()
    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM analyses WHERE profile = ? ORDER BY id DESC",
                            (profile,)).fetchall()
    return [dict(r) for r in rows]


def get_analysis(analysis_id):
    _require_history()
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["missing"] = json.loads(data["missing_json"] or "[]")
    data["plan"] = json.loads(data["plan_json"] or "[]")
    return data


def delete_analysis(analysis_id):
    _require_history()
    with closing(connect()) as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.execute("DELETE FROM progress WHERE analysis_id = ?", (analysis_id,))
        conn.commit()
    log.info("Deleted analysis %d", analysis_id)


def set_progress(analysis_id, skill, done):
    _require_history()
    with closing(connect()) as conn:
        conn.execute("INSERT INTO progress (analysis_id, skill, done) VALUES (?,?,?) "
                     "ON CONFLICT(analysis_id, skill) DO UPDATE SET done = excluded.done",
                     (analysis_id, skill, int(done)))
        conn.commit()


def get_progress(analysis_id):
    _require_history()
    with closing(connect()) as conn:
        rows = conn.execute("SELECT skill, done FROM progress WHERE analysis_id = ?",
                            (analysis_id,)).fetchall()
    return {r["skill"]: bool(r["done"]) for r in rows}


def stats(profile):
    _require_history()
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, AVG(match_pct) AS match, AVG(strength) AS strength, "
            "AVG(weeks) AS weeks FROM analyses WHERE profile = ?", (profile,)).fetchone()
        total = conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"]
    return {"count": row["n"] or 0, "match": row["match"] or 0.0,
            "strength": row["strength"] or 0.0, "weeks": row["weeks"] or 0.0, "total": total}