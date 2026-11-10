import re
import time
from datetime import date, timedelta
import numpy as np
import pandas as pd
from core import config as cfg
from core import nlp_feedback as nlp
from core.models import artifacts_exist, train_all, load_artifacts
from core.preprocess import clean_text, extract_skills, count_skill_mentions, skill_features
from core.prerequisites import learning_order
from core.logger_setup import get_logger

log = get_logger("engine")


class CareerEngine:
    def __init__(self, bundle, nn):
        self.b = bundle
        self.nn = nn
        self.jobs = bundle["jobs"]
        self.tfidf = bundle["tfidf"]
        self.J = self.tfidf.transform(self.jobs["clean"])
        self.job_titles = self.jobs["title"].tolist()
        self.job_domains = self.jobs["domain"].tolist()
        self.job_skills = self.jobs["job_skills"].tolist()

    @classmethod
    def load_or_train(cls):
        if not artifacts_exist():
            log.info("Artifacts not found, training models")
            train_all()
        bundle, nn = load_artifacts()
        log.info("Engine ready: %d jobs, %d skills", len(bundle["jobs"]), len(cfg.ALL_SKILLS))
        return cls(bundle, nn)

    def predict(self, feats):
        row = pd.DataFrame([feats])
        reg = self.b["reg"]
        clf = self.b["clf"]
        weeks = float(reg["best"].predict(reg["scaler"].transform(row[cfg.REG_FEATURES]))[0])
        level = clf["best"].predict(clf["scaler"].transform(row[cfg.CLF_FEATURES]))[0]
        return str(level), max(0.0, weeks)

    def language_score(self, text):
        x = self.b["nn_vec"].transform([clean_text(text)]).toarray().astype("float32")
        return float(self.nn(x, training=False).numpy()[0][0])

    def domain_coverage(self, cand_skills):
        return {d: len(set(cand_skills) & set(s)) / len(s) for d, s in cfg.DOMAIN_SKILLS.items()}

    def candidate_vector(self, cand_skills):
        vec = np.zeros(len(cfg.ALL_SKILLS))
        for s in cand_skills:
            vec[cfg.ALL_SKILLS.index(s)] = 1
        return vec

    def candidate_point(self, cand_skills):
        vec = self.candidate_vector(cand_skills).reshape(1, -1)
        return self.b["pca"]["pca"].transform(vec)[0]

    def alternate_careers(self, clean, cand_skills, years):
        vec = self.candidate_vector(cand_skills)
        cents = self.b["pca"]["centroids"]
        sims = cents @ vec / (np.linalg.norm(cents, axis=1) * (np.linalg.norm(vec) + 1e-9) + 1e-9)
        top = np.argsort(sims)[::-1][:3]
        cand_vec = self.tfidf.transform([clean])
        job_sims = (self.J @ cand_vec.T).toarray().ravel()
        rows = []
        for t in top:
            d = cfg.DOMAIN_NAMES[t]
            ids = [i for i, dom in enumerate(self.job_domains) if dom == d]
            best = max(ids, key=lambda i: job_sims[i])
            feats = skill_features(cand_skills, self.job_skills[best], years)
            feats["match_score"] = float(job_sims[best])
            level, weeks = self.predict(feats)
            rows.append({"domain": d, "fit": float(sims[t]) * 100, "job": self.job_titles[best],
                         "job_id": best, "level": level, "weeks": weeks})
        return rows

    def skill_levels(self, clean):
        rows = []
        for s, n in count_skill_mentions(clean).items():
            level = "Beginner" if n == 1 else ("Intermediate" if n == 2 else "Advanced")
            rows.append({"skill": s, "mentions": n, "level": level, "difficulty": cfg.DIFFICULTY[s]})
        return pd.DataFrame(rows, columns=["skill", "mentions", "level", "difficulty"])

    def match_for(self, clean, job_id):
        return float(self.tfidf.transform([clean]).multiply(self.J[job_id]).sum())

    def weeks_per_base_unit(self, missing, ml_weeks, hours):
        base = sum(cfg.SKILL_WEEKS[m] for m in missing)
        if base == 0:
            return 0.0
        return max(ml_weeks, 1.0) / base * cfg.HOURS_BASELINE / max(hours, 1)

    def learning_weeks(self, order, result, hours):
        unit = self.weeks_per_base_unit(result["missing"], result["weeks"], hours)
        return sum(cfg.SKILL_WEEKS[s] for s in order) * unit

    def timeline(self, missing, owned, ml_weeks, hours):
        cols = ["skill", "type", "difficulty", "start", "weeks", "end"]
        if not missing:
            return pd.DataFrame(columns=cols)
        unit = self.weeks_per_base_unit(missing, ml_weeks, hours)
        start = 0.0
        rows = []
        for s in learning_order(missing, owned):
            w = cfg.SKILL_WEEKS[s] * unit
            rows.append({"skill": s, "type": "target skill" if s in missing else "foundation",
                         "difficulty": cfg.DIFFICULTY[s], "start": start, "weeks": w, "end": start + w})
            start += w
        return pd.DataFrame(rows, columns=cols)

    def plan_for_save(self, result):
        tl = self.timeline(result["missing"], result["cand_skills"], result["weeks"], cfg.HOURS_BASELINE)
        return [{"skill": r.skill, "weeks": round(float(r.weeks), 2), "foundation": r.type == "foundation"}
                for r in tl.itertuples()]

    @staticmethod
    def parse_years(text):
        m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)", text.lower())
        return min(int(m.group(1)), 20) if m else 0

    def screen(self, resume_text, years, job_id):
        clean = clean_text(resume_text)
        cand = extract_skills(clean)
        match = self.match_for(clean, job_id)
        feats = skill_features(cand, self.job_skills[job_id], years)
        feats["match_score"] = match
        level, weeks = self.predict(feats)
        fit = 0.6 * feats["skill_match_ratio"] + 0.4 * min(match / 0.25, 1.0)
        missing = sorted(set(self.job_skills[job_id]) - set(cand))
        return {"fit_score": round(fit * 100, 1), "skill_match_pct": round(feats["skill_match_ratio"] * 100, 1),
                "text_match_pct": round(match * 100, 1), "level": level, "weeks_to_ready": round(weeks, 1),
                "years": years, "matched": sorted(set(self.job_skills[job_id]) & set(cand)),
                "missing": missing}

    def analyze(self, resume_text, years, job_id):
        t0 = time.time()
        clean = clean_text(resume_text)
        cand_skills = extract_skills(clean)
        job_skills = self.job_skills[job_id]
        match = self.match_for(clean, job_id)
        feats = skill_features(cand_skills, job_skills, years)
        feats["match_score"] = match
        level, weeks = self.predict(feats)
        missing = sorted(set(job_skills) - set(cand_skills), key=lambda s: (-cfg.DIFFICULTY[s], s))
        matched = sorted(set(job_skills) & set(cand_skills))
        lang_prob = self.language_score(resume_text)
        analysis = nlp.analyze_text(resume_text, job_skills, cand_skills)
        result = {
            "job_id": job_id, "job_title": self.job_titles[job_id], "domain": self.job_domains[job_id],
            "job_skills": sorted(job_skills), "cand_skills": sorted(cand_skills),
            "matched": matched, "missing": missing, "match_pct": match * 100,
            "skill_match_pct": feats["skill_match_ratio"] * 100, "level": level, "weeks": weeks,
            "weeks_by_skill": {s: cfg.SKILL_WEEKS[s] for s in missing}, "lang_prob": lang_prob,
            "nlp": analysis, "strength": nlp.strength_score(analysis, lang_prob),
            "alternates": self.alternate_careers(clean, cand_skills, years),
            "coverage": self.domain_coverage(cand_skills),
            "skill_levels": self.skill_levels(clean), "years": years, "resume_text": resume_text,
            "feats": feats,
        }
        log.info("Analyzed resume for '%s' | match=%.1f%% level=%s weeks=%.1f (%.2fs)",
                 result["job_title"], result["match_pct"], level, weeks, time.time() - t0)
        return result

    def rank_jobs(self, resume_text, years, top_n=10, domain="All"):
        clean = clean_text(resume_text)
        cand = extract_skills(clean)
        cand_vec = self.tfidf.transform([clean])
        sims = (self.J @ cand_vec.T).toarray().ravel()
        scored = []
        for idx in range(len(self.job_titles)):
            if domain != "All" and self.job_domains[idx] != domain:
                continue
            js = set(self.job_skills[idx])
            ratio = len(set(cand) & js) / max(len(js), 1)
            fit = 0.6 * ratio + 0.4 * min(sims[idx] / 0.25, 1.0)
            scored.append((fit, idx, ratio))
        scored.sort(reverse=True)
        rows = []
        for fit, idx, ratio in scored[:top_n]:
            feats = skill_features(cand, self.job_skills[idx], years)
            feats["match_score"] = float(sims[idx])
            level, weeks = self.predict(feats)
            js = set(self.job_skills[idx])
            rows.append({
                "job_id": idx, "title": self.job_titles[idx], "domain": self.job_domains[idx],
                "fit_score": round(fit * 100, 1), "skill_match_pct": round(ratio * 100, 1),
                "level": level, "weeks_to_ready": round(weeks, 1),
                "matched": sorted(js & set(cand)), "missing": sorted(js - set(cand)),
            })
        log.info("Ranked %d jobs (domain=%s)", len(rows), domain)
        return pd.DataFrame(rows)

    def report_text(self, result):
        lines = [
            "CAREERPATH AI REPORT",
            f"Target job: {result['job_title']}",
            f"Match score: {result['match_pct']:.1f}%",
            f"Skill match: {result['skill_match_pct']:.1f}%",
            f"Readiness level: {result['level']}",
            f"Estimated time to job ready: {result['weeks']:.1f} weeks",
            f"Resume strength: {result['strength']}/100",
            "",
            "Matched skills: " + (", ".join(result["matched"]) or "None"),
            "Missing skills: " + (", ".join(result["missing"]) or "None"),
            "",
            "Resume feedback:",
        ]
        lines += ["- " + t for t in result["nlp"]["tips"]]
        lines += ["", "Alternate careers:"]
        for a in result["alternates"]:
            lines.append(f"- {a['domain']} (fit {a['fit']:.0f}%) | best job: {a['job']} | "
                         f"level: {a['level']} | {a['weeks']:.1f} weeks")
        ready = date.today() + timedelta(weeks=result["weeks"])
        lines += ["", f"Estimated ready date at 10 hours per week: {ready.isoformat()}"]
        return "\n".join(lines)