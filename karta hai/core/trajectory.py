import pandas as pd

from core import config as cfg
from core.preprocess import clean_text, skill_features
from core.prerequisites import learning_order


TRAJECTORY_COLUMNS = ["week", "skill", "skill_match_pct", "match_pct", "level", "weeks_left"]
ROI_COLUMNS = ["skill", "roi", "match_gain_pct", "cost_weeks", "weeks_saved", "new_level", "foundations"]


def _evaluate(engine, resume_text, years, job_id, owned, learned=()):
    skills = set(owned) | set(learned)
    text = clean_text(f"{resume_text} {' '.join(learned)}")
    features = skill_features(skills, engine.job_skills[job_id], years)
    features["match_score"] = engine.match_for(text, job_id)
    level, weeks = engine.predict(features)
    return {
        "skill_match_pct": features["skill_match_ratio"] * 100,
        "match_pct": features["match_score"] * 100,
        "level": level,
        "weeks_left": weeks,
    }


def build_trajectory(engine, resume_text, years, job_id, owned, timeline):
    if timeline.empty:
        return pd.DataFrame(columns=TRAJECTORY_COLUMNS)

    rows = []
    learned = []
    for row in timeline.itertuples():
        learned.append(row.skill)
        prediction = _evaluate(engine, resume_text, years, job_id, owned, learned)
        rows.append({
            "week": float(row.end),
            "skill": row.skill,
            **prediction,
        })
    return pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS)


def level_changes(trajectory):
    if trajectory.empty:
        return trajectory.copy()
    changed = trajectory["level"].ne(trajectory["level"].shift())
    changed.iloc[0] = False
    return trajectory.loc[changed].copy()


def skill_roi(engine, resume_text, years, job_id, owned, missing):
    baseline = _evaluate(engine, resume_text, years, job_id, owned)
    rows = []
    for skill in missing:
        order = learning_order([skill], owned)
        foundations = [item for item in order if item not in owned and item != skill]
        cost = sum(cfg.SKILL_WEEKS[item] for item in order)
        after = _evaluate(engine, resume_text, years, job_id, owned, order)
        gain = max(0.0, after["skill_match_pct"] - baseline["skill_match_pct"])
        rows.append({
            "skill": skill,
            "roi": gain / cost if cost else 0.0,
            "match_gain_pct": gain,
            "cost_weeks": cost,
            "weeks_saved": max(0.0, baseline["weeks_left"] - after["weeks_left"]),
            "new_level": after["level"],
            "foundations": ", ".join(foundations),
        })
    rows.sort(key=lambda row: (-row["roi"], -row["match_gain_pct"], row["skill"]))
    return pd.DataFrame(rows, columns=ROI_COLUMNS)


def what_if(engine, result, chosen):
    owned = result["cand_skills"]
    requested = list(dict.fromkeys(chosen))
    invalid = set(requested) - set(cfg.ALL_SKILLS)
    if invalid:
        raise ValueError(f"Unknown skills: {', '.join(sorted(invalid))}")

    order = learning_order(requested, owned)
    foundations = [skill for skill in order if skill not in owned and skill not in requested]
    return {
        "before": _evaluate(engine, result["resume_text"], result["years"], result["job_id"], owned),
        "after": _evaluate(engine, result["resume_text"], result["years"], result["job_id"], owned, order),
        "order": order,
        "foundations": foundations,
    }