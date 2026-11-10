import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from core import config as cfg

SKILL_PATTERNS = {s: re.compile(r"\b" + re.escape(s) + r"\b") for s in cfg.ALL_SKILLS}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def extract_skills(clean):
    return [s for s, p in SKILL_PATTERNS.items() if p.search(clean)]


def count_skill_mentions(clean):
    counts = {}
    for s, p in SKILL_PATTERNS.items():
        n = len(p.findall(clean))
        if n > 0:
            counts[s] = n
    return counts


def skill_features(resume_skills, job_skills, years):
    rs = set(resume_skills)
    js = set(job_skills)
    matched = rs & js
    missing = js - rs
    req_n = max(len(js), 1)
    avg_diff = float(np.mean([cfg.DIFFICULTY[s] for s in matched])) if matched else 0.0
    return {
        "matched_n": len(matched),
        "missing_n": len(missing),
        "skill_match_ratio": len(matched) / req_n,
        "missing_difficulty": sum(cfg.DIFFICULTY[s] for s in missing),
        "missing_weeks": sum(cfg.SKILL_WEEKS[s] for s in missing),
        "avg_diff_matched": avg_diff,
        "n_resume_skills": len(rs),
        "years_exp": years,
    }


def prepare_text(resumes, jobs):
    resumes["clean"] = resumes["resume_text"].apply(clean_text)
    jobs["clean"] = jobs["job_text"].apply(clean_text)
    resumes["skills"] = resumes["clean"].apply(extract_skills)
    jobs["job_skills"] = jobs["clean"].apply(extract_skills)
    return resumes, jobs


def fit_tfidf(resumes, jobs):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf.fit(pd.concat([resumes["clean"], jobs["clean"]]))
    return tfidf


def build_pairs(resumes, jobs, tfidf, rng):
    R = tfidf.transform(resumes["clean"])
    J = tfidf.transform(jobs["clean"])
    job_idx = resumes["target_job_id"].to_numpy()
    match_scores = np.asarray(R.multiply(J[job_idx]).sum(axis=1)).ravel()

    rows = []
    for i in range(len(resumes)):
        rows.append(skill_features(resumes["skills"].iloc[i],
                                   jobs["job_skills"].iloc[int(job_idx[i])],
                                   int(resumes["years_exp"].iloc[i])))
    pairs = pd.DataFrame(rows)
    pairs["match_score"] = match_scores
    pairs["domain"] = resumes["domain"].to_numpy()

    exp_factor = np.clip(1.25 - 0.05 * pairs["years_exp"].to_numpy(), 0.6, 1.25)
    pairs["time_to_ready"] = np.maximum(
        0, pairs["missing_weeks"].to_numpy() * exp_factor + rng.normal(0, 1.5, len(pairs)))

    score = (0.6 * pairs["skill_match_ratio"].to_numpy()
             + 0.4 * np.minimum(pairs["years_exp"].to_numpy() / 10, 1)
             + rng.normal(0, 0.05, len(pairs)))
    pairs["level"] = pd.cut(score, bins=[-np.inf, 0.45, 0.70, np.inf],
                            labels=["Junior", "Mid", "Senior"]).astype(str)
    return pairs, J