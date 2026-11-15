import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from core import config as cfg
from core.taxonomy import SKILL_ALIASES, SOFT_SKILLS

SKILL_PATTERNS = {s: re.compile(r"\b" + re.escape(s) + r"\b") for s in cfg.ALL_SKILLS}
ALIAS_PATTERNS = {a: (re.compile(r"\b" + re.escape(a) + r"\b"), target)
                  for a, target in SKILL_ALIASES.items() if target in cfg.ALL_SKILLS}
SOFT_PATTERNS = {s: re.compile(r"\b" + re.escape(s) + r"\b") for s in SOFT_SKILLS}


def clean_text(text: str) -> str:
    """Preprocesses raw text by lower-casing, removing punctuation, and filtering stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def extract_skills(clean: str) -> list:
    """
    Extracts canonical skills from cleaned text using direct pattern matching
    and alias resolution. Returns list of unique canonical skills.
    """
    found = set()
    # 1. Direct canonical skills
    for s, p in SKILL_PATTERNS.items():
        if p.search(clean):
            found.add(s)
            
    # 2. Aliases and synonym mappings
    for alias, (p, target) in ALIAS_PATTERNS.items():
        if p.search(clean):
            found.add(target)
            
    return sorted(found)


def extract_soft_skills(clean: str) -> list:
    """Extracts soft skills and professional competencies from cleaned text."""
    found = set()
    for s, p in SOFT_PATTERNS.items():
        if p.search(clean):
            found.add(s)
    return sorted(found)


def count_skill_mentions(clean: str) -> dict:
    """
    Counts total mentions for each matched skill, combining canonical
    and alias occurrences.
    """
    counts = {}
    for s, p in SKILL_PATTERNS.items():
        n = len(p.findall(clean))
        if n > 0:
            counts[s] = counts.get(s, 0) + n
            
    for alias, (p, target) in ALIAS_PATTERNS.items():
        n = len(p.findall(clean))
        if n > 0:
            counts[target] = counts.get(target, 0) + n
            
    return counts


def skill_features(resume_skills, job_skills, years):
    """
    Calculates numerical features for regression and classification models.
    Preserves exact keys expected by cfg.REG_FEATURES and cfg.CLF_FEATURES.
    """
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
