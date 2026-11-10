import re
import pandas as pd
from core import config as cfg
from core.logger_setup import get_logger
from core.nlp_feedback import STRONG_REGEX, WEAK_REGEX
from core.preprocess import clean_text, extract_skills

log = get_logger("data")

TEXT_COLUMNS = ["Resume_str", "Resume", "resume_text", "text", "Text"]

def generate_jobs(rng):
    rows = []
    job_id = 0
    for d in cfg.DOMAIN_NAMES:
        for _ in range(cfg.JOBS_PER_DOMAIN):
            n_req = int(rng.integers(6, 9))
            req = list(rng.choice(cfg.DOMAIN_SKILLS[d], size=n_req, replace=False))
            if d == "Java Developer" and "java" not in req:
                req[0] = "java"
            title = f"{cfg.JOB_LEVELS[int(rng.integers(0, 3))]} {d}"
            text = (f"{title} position. We are hiring a {title}. The candidate must have strong "
                    f"knowledge of {', '.join(req)}. Responsibilities include creating and "
                    f"maintaining solutions for the team and clients.")
            rows.append({"job_id": job_id, "title": title, "domain": d, "job_text": text})
            job_id += 1
    return pd.DataFrame(rows)


def generate_resumes(rng):
    rows = []
    for _ in range(cfg.N_RESUMES):
        d = cfg.DOMAIN_NAMES[int(rng.integers(0, len(cfg.DOMAIN_NAMES)))]
        years = int(rng.integers(0, 13))
        k = int(rng.integers(3, 9))
        own = list(rng.choice(cfg.DOMAIN_SKILLS[d], size=k, replace=False))
        other_pool = [s for s in cfg.ALL_SKILLS if s not in cfg.DOMAIN_SKILLS[d]]
        n_noise = int(rng.integers(0, 3))
        noise = list(rng.choice(other_pool, size=n_noise, replace=False)) if n_noise > 0 else []
        skills = own + noise
        p_strong = rng.beta(2, 2)
        n_sent = int(rng.integers(4, 8))
        sentences = []
        n_strong = 0
        n_weak = 0
        for _ in range(n_sent):
            skill = skills[int(rng.integers(0, len(skills)))]
            thing = cfg.THINGS[int(rng.integers(0, len(cfg.THINGS)))]
            if rng.random() < p_strong:
                verb = cfg.STRONG_VERBS[int(rng.integers(0, len(cfg.STRONG_VERBS)))]
                sentences.append(f"{verb} {thing} using {skill}")
                n_strong += 1
            else:
                phrase = cfg.WEAK_PHRASES[int(rng.integers(0, len(cfg.WEAK_PHRASES)))]
                sentences.append(f"{phrase} {thing} using {skill}")
                n_weak += 1
        edu = cfg.EDUCATION_LIST[int(rng.integers(0, len(cfg.EDUCATION_LIST)))]
        text = (f"{edu} with {years} years of experience. Technical skills include "
                f"{', '.join(skills)}. " + ". ".join(sentences) + ".")
        rows.append({"domain": d, "years_exp": years, "resume_text": text,
                     "n_strong": n_strong, "n_weak": n_weak,
                     "language_label": int(n_strong > n_weak), "source": "synthetic"})
    return pd.DataFrame(rows)


def assign_targets(resumes, jobs, rng):
    jobs_by_domain = {d: jobs.loc[jobs["domain"] == d, "job_id"].to_numpy()
                      for d in cfg.DOMAIN_NAMES}
    targets = []
    for d in resumes["domain"]:
        if rng.random() < 0.75:
            targets.append(int(rng.choice(jobs_by_domain[d])))
        else:
            targets.append(int(rng.integers(0, len(jobs))))
    resumes["target_job_id"] = targets
    return resumes


def load_external_resumes(path):
    raw = pd.read_csv(path)
    col = next((c for c in TEXT_COLUMNS if c in raw.columns), None)
    if col is None:
        raise ValueError("CSV needs one of these text columns: " + ", ".join(TEXT_COLUMNS))
    rows = []
    for text in raw[col].dropna().astype(str):
        if len(text.split()) < 30:
            continue
        skills = extract_skills(clean_text(text))
        if len(skills) < 2:
            continue
        coverage = {d: len(set(skills) & set(s)) / len(s) for d, s in cfg.DOMAIN_SKILLS.items()}
        domain = max(coverage, key=coverage.get)
        m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)", text.lower())
        years = min(int(m.group(1)), 20) if m else 2
        rows.append({"domain": domain, "years_exp": years, "resume_text": text,
                     "n_strong": len(STRONG_REGEX.findall(text)),
                     "n_weak": len(WEAK_REGEX.findall(text)), "source": "real"})
    ext = pd.DataFrame(rows)
    if ext.empty:
        raise ValueError("No usable resumes found in the CSV.")
    diff = ext["n_strong"] - ext["n_weak"]
    label = (diff > diff.median()).astype(int)
    if label.nunique() < 2:
        label = (diff >= diff.median()).astype(int)
    ext["language_label"] = label
    log.info("Loaded %d usable real resumes from %s", len(ext), path)
    return ext


def generate_datasets(rng, resume_csv=None, only_real=False):
    jobs = generate_jobs(rng)
    resumes = generate_resumes(rng)
    if resume_csv:
        real = load_external_resumes(resume_csv)
        resumes = real if only_real else pd.concat([resumes, real], ignore_index=True)
    resumes = assign_targets(resumes, jobs, rng)
    resumes.insert(0, "resume_id", range(len(resumes)))
    skill_table = pd.DataFrame({
        "skill": cfg.ALL_SKILLS,
        "difficulty": [cfg.DIFFICULTY[s] for s in cfg.ALL_SKILLS],
        "learning_weeks": [cfg.SKILL_WEEKS[s] for s in cfg.ALL_SKILLS],
    })
    resumes.to_csv(cfg.DATA_DIR / "resumes_dataset.csv", index=False)
    jobs.to_csv(cfg.DATA_DIR / "jobs_dataset.csv", index=False)
    skill_table.to_csv(cfg.DATA_DIR / "skill_reference.csv", index=False)
    log.info("Prepared %d resumes (%s), %d jobs, %d skills", len(resumes),
             resumes["source"].value_counts().to_dict(), len(jobs), len(skill_table))
    return resumes, jobs, skill_table