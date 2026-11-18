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
    filename TEXT DEFAULT 'Resume Document',
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
        # Migrate existing tables if filename column is absent
        cur = conn.execute("PRAGMA table_info(analyses)")
        cols = [r[1] for r in cur.fetchall()]
        if "filename" not in cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN filename TEXT DEFAULT 'Resume Document'")
        conn.commit()


def save_analysis(profile, result, plan, filename=None):
    _require_history()
    doc_fn = filename or result.get("metadata", {}).get("filename") or "Resume Document"
    with closing(connect()) as conn:
        cur = conn.execute("PRAGMA table_info(analyses)")
        cols = [r[1] for r in cur.fetchall()]
        if "filename" in cols:
            cur = conn.execute(
                "INSERT INTO analyses (created_at, profile, filename, job_id, job_title, domain, match_pct, "
                "skill_match_pct, level, weeks, lang_score, strength, years, resume_text, "
                "missing_json, plan_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), profile, doc_fn, result["job_id"],
                 result["job_title"], result["domain"], result["match_pct"],
                 result["skill_match_pct"], result["level"], result["weeks"],
                 result["lang_prob"] * 100, result["strength"], result["years"],
                 result["resume_text"], json.dumps(result["missing"]), json.dumps(plan)))
        else:
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
    log.info("Saved analysis %d for profile '%s' (file: %s)", analysis_id, profile, doc_fn)
    return analysis_id


def list_analyses(profile=None):
    _require_history()
    with closing(connect()) as conn:
        if profile:
            rows = conn.execute("SELECT * FROM analyses WHERE profile = ? ORDER BY id DESC",
                                (profile,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if "filename" not in d or not d["filename"]:
            d["filename"] = f"Resume Document #{d['id']}"
        results.append(d)
    return results


def get_analysis(analysis_id):
    _require_history()
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    if "filename" not in data or not data["filename"]:
        data["filename"] = f"Resume Document #{data['id']}"
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


def clear_all_analyses(profile=None):
    _require_history()
    with closing(connect()) as conn:
        if profile:
            cur = conn.execute("SELECT id FROM analyses WHERE profile = ?", (profile,))
            ids = [r[0] for r in cur.fetchall()]
            conn.execute("DELETE FROM analyses WHERE profile = ?", (profile,))
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM progress WHERE analysis_id IN ({placeholders})", ids)
        else:
            conn.execute("DELETE FROM analyses")
            conn.execute("DELETE FROM progress")
        conn.commit()
    log.info("Cleared analyses history for profile: %s", profile or "ALL")


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


def stats(profile=None):
    _require_history()
    with closing(connect()) as conn:
        if profile:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(match_pct) AS match, AVG(skill_match_pct) AS skill_match, "
                "AVG(strength) AS strength, AVG(weeks) AS weeks FROM analyses WHERE profile = ?", (profile,)).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(match_pct) AS match, AVG(skill_match_pct) AS skill_match, "
                "AVG(strength) AS strength, AVG(weeks) AS weeks FROM analyses").fetchone()
        total = conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"]
    return {
        "count": row["n"] or 0,
        "match": row["match"] or 0.0,
        "skill_match": row["skill_match"] or 0.0,
        "strength": row["strength"] or 0.0,
        "weeks": row["weeks"] or 0.0,
        "total": total,
    }