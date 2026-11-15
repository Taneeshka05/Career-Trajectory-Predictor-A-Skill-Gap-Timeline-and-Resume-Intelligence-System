import re
import time
from datetime import date, timedelta
import numpy as np
import pandas as pd
from core import config as cfg
from core import nlp_feedback as nlp
from core.models import artifacts_exist, train_all, load_artifacts
from core.preprocess import (clean_text, extract_skills, extract_soft_skills,
                             count_skill_mentions, skill_features)
from core.prerequisites import learning_order
from core.taxonomy import (find_transferable_credits, get_skill_category,
                           categorize_skills, SKILL_TAXONOMY, SOFT_SKILLS)
from core.parser import extract_resume_metadata, is_valid_resume
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
        self._require_valid_resume(text)
        x = self.b["nn_vec"].transform([clean_text(text)]).toarray().astype("float32")
        return float(self.nn(x, training=False).numpy()[0][0])

    @staticmethod
    def _require_valid_resume(text):
        if not is_valid_resume(text):
            raise ValueError(
                "Input rejected: the document does not contain enough structured resume content."
            )

    def domain_coverage(self, cand_skills):
        return {d: len(set(cand_skills) & set(s)) / len(s) for d, s in cfg.DOMAIN_SKILLS.items()}

    def candidate_vector(self, cand_skills):
        vec = np.zeros(len(cfg.ALL_SKILLS))
        for s in cand_skills:
            if s in cfg.ALL_SKILLS:
                vec[cfg.ALL_SKILLS.index(s)] = 1
        return vec

    def candidate_point(self, cand_skills):
        vec = self.candidate_vector(cand_skills).reshape(1, -1)
        return self.b["pca"]["pca"].transform(vec)[0]

    def alternate_careers(self, clean, cand_skills, years):
        vec = self.candidate_vector(cand_skills)
        cents = self.b["pca"]["centroids"]
        sims = cents @ vec / (np.linalg.norm(cents, axis=1) * (np.linalg.norm(vec) + 1e-9) + 1e-9)
        top = np.argsort(sims)[::-1][:4]
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
            
            domain_skills = set(cfg.DOMAIN_SKILLS[d])
            overlap = sorted(set(cand_skills) & domain_skills)
            missing = sorted(domain_skills - set(cand_skills))
            transferable = find_transferable_credits(set(cand_skills), set(missing))
            
            rationale = (
                f"You hold {len(overlap)} of {len(domain_skills)} core {d} skills. "
                f"{len(transferable)} additional skills have high cross-domain transferability."
            )
            rows.append({
                "domain": d,
                "fit": float(sims[t]) * 100,
                "job": self.job_titles[best],
                "job_id": best,
                "level": level,
                "weeks": weeks,
                "overlap": overlap,
                "missing": missing,
                "transferable": transferable,
                "rationale": rationale
            })
        return rows

    def skill_levels(self, clean):
        rows = []
        mentions = count_skill_mentions(clean)
        for s, n in mentions.items():
            level = "Beginner" if n == 1 else ("Intermediate" if n == 2 else "Advanced")
            diff = cfg.DIFFICULTY.get(s, 2)
            cat = get_skill_category(s)
            rows.append({
                "skill": s,
                "category": cat,
                "mentions": n,
                "level": level,
                "difficulty": diff,
                "evidence_type": "Resume Text Keyword Mentions"
            })
        return pd.DataFrame(rows, columns=["skill", "category", "mentions", "level", "difficulty", "evidence_type"])

    def match_for(self, clean, job_id):
        return float(self.tfidf.transform([clean]).multiply(self.J[job_id]).sum())

    def weeks_per_base_unit(self, missing, ml_weeks, hours):
        base = sum(cfg.SKILL_WEEKS.get(m, 5) for m in missing)
        if base == 0:
            return 0.0
        return max(ml_weeks, 1.0) / base * cfg.HOURS_BASELINE / max(hours, 1)

    def learning_weeks(self, order, result, hours):
        unit = self.weeks_per_base_unit(result["missing"], result["weeks"], hours)
        return sum(cfg.SKILL_WEEKS.get(s, 5) for s in order) * unit

    def timeline(self, missing, owned, ml_weeks, hours):
        cols = ["skill", "type", "difficulty", "start", "weeks", "end"]
        if not missing:
            return pd.DataFrame(columns=cols)
        unit = self.weeks_per_base_unit(missing, ml_weeks, hours)
        start = 0.0
        rows = []
        for s in learning_order(missing, owned):
            w = cfg.SKILL_WEEKS.get(s, 5) * unit
            rows.append({
                "skill": s,
                "type": "target skill" if s in missing else "foundation",
                "difficulty": cfg.DIFFICULTY.get(s, 2),
                "start": start,
                "weeks": w,
                "end": start + w
            })
            start += w
        return pd.DataFrame(rows, columns=cols)

    def plan_for_save(self, result):
        tl = self.timeline(result["missing"], result["cand_skills"], result["weeks"], cfg.HOURS_BASELINE)
        return [{"skill": r.skill, "weeks": round(float(r.weeks), 2), "foundation": r.type == "foundation"}
                for r in tl.itertuples()]

    @staticmethod
    def parse_years(text):
        return extract_resume_metadata(text)["extracted_years"]

    def screen(self, resume_text, years, job_id):
        self._require_valid_resume(resume_text)
        years = extract_resume_metadata(resume_text)["extracted_years"]
        clean = clean_text(resume_text)
        cand = extract_skills(clean)
        match = self.match_for(clean, job_id)
        feats = skill_features(cand, self.job_skills[job_id], years)
        feats["match_score"] = match
        level, weeks = self.predict(feats)
        fit = 0.6 * feats["skill_match_ratio"] + 0.4 * min(match / 0.25, 1.0)
        missing = sorted(set(self.job_skills[job_id]) - set(cand))
        matched = sorted(set(self.job_skills[job_id]) & set(cand))
        transferable = find_transferable_credits(set(cand), set(missing))
        return {
            "fit_score": round(fit * 100, 1),
            "skill_match_pct": round(feats["skill_match_ratio"] * 100, 1),
            "text_match_pct": round(match * 100, 1),
            "level": level,
            "weeks_to_ready": round(weeks, 1),
            "years": years,
            "matched": matched,
            "missing": missing,
            "transferable": transferable,
        }

    def analyze(self, resume_text, years, job_id):
        self._require_valid_resume(resume_text)
        metadata = extract_resume_metadata(resume_text)
        years = metadata["extracted_years"]
        t0 = time.time()
        clean = clean_text(resume_text)
        cand_skills = extract_skills(clean)
        cand_soft = extract_soft_skills(clean)
        job_skills = self.job_skills[job_id]
        
        job_clean = self.jobs.loc[self.jobs["job_id"] == job_id, "clean"].iloc[0]
        job_soft = extract_soft_skills(job_clean)
        
        match = self.match_for(clean, job_id)
        feats = skill_features(cand_skills, job_skills, years)
        feats["match_score"] = match
        level, weeks = self.predict(feats)
        
        missing = sorted(set(job_skills) - set(cand_skills), key=lambda s: (-cfg.DIFFICULTY.get(s, 2), s))
        matched = sorted(set(job_skills) & set(cand_skills))
        
        # Transferable skills analysis
        transferable = find_transferable_credits(set(cand_skills), set(missing))
        transferable_gain = sum(c["credit"] for c in transferable) / max(len(job_skills), 1)
        transferable_match_pct = min(100.0, (feats["skill_match_ratio"] + transferable_gain) * 100)
        
        # Soft skills alignment
        soft_overlap = sorted(set(cand_soft) & set(job_soft)) if job_soft else sorted(cand_soft)
        soft_match_pct = (len(soft_overlap) / max(len(job_soft), 1) * 100) if job_soft else (80.0 if cand_soft else 50.0)
        
        # Experience alignment
        req_exp = 1 if "junior" in self.job_titles[job_id].lower() else (3 if "mid" in self.job_titles[job_id].lower() else 6)
        exp_fit_ratio = min(years / max(req_exp, 1), 1.2)
        
        # Composite Explainable Compatibility Score (0 - 100)
        norm_text = min(match / 0.25, 1.0)
        composite_compat = (
            0.40 * feats["skill_match_ratio"] +
            0.15 * min(transferable_gain, 0.35) +
            0.20 * norm_text +
            0.15 * (soft_match_pct / 100.0) +
            0.10 * min(exp_fit_ratio, 1.0)
        ) * 100.0
        composite_compat = round(float(np.clip(composite_compat, 5.0, 99.0)), 1)
        
        # Detailed gap explanations
        gap_explanations = []
        for s in missing:
            diff = cfg.DIFFICULTY.get(s, 2)
            cat = get_skill_category(s)
            base_w = cfg.SKILL_WEEKS.get(s, 5)
            gap_explanations.append({
                "skill": s,
                "category": cat,
                "difficulty": diff,
                "base_weeks": base_w,
                "hours": base_w * 10,
                "priority": "Core Requirement" if s in job_skills[:5] else "Secondary Requirement",
                "explanation": f"Important {cat} capability for {self.job_titles[job_id]} tasks and architecture."
            })
            
        lang_prob = self.language_score(resume_text)
        analysis = nlp.analyze_text(resume_text, job_skills, cand_skills)
        result = {
            "job_id": job_id,
            "job_title": self.job_titles[job_id],
            "domain": self.job_domains[job_id],
            "job_skills": sorted(job_skills),
            "cand_skills": sorted(cand_skills),
            "cand_soft_skills": sorted(cand_soft),
            "job_soft_skills": sorted(job_soft),
            "soft_overlap": soft_overlap,
            "soft_match_pct": round(soft_match_pct, 1),
            "matched": matched,
            "missing": missing,
            "transferable": transferable,
            "transferable_match_pct": round(transferable_match_pct, 1),
            "composite_compat": composite_compat,
            "gap_explanations": gap_explanations,
            "match_pct": match * 100,
            "skill_match_pct": feats["skill_match_ratio"] * 100,
            "level": level,
            "weeks": weeks,
            "weeks_by_skill": {s: cfg.SKILL_WEEKS.get(s, 5) for s in missing},
            "lang_prob": lang_prob,
            "nlp": analysis,
            "metadata": metadata,
            "strength": nlp.strength_score(analysis, lang_prob),
            "alternates": self.alternate_careers(clean, cand_skills, years),
            "coverage": self.domain_coverage(cand_skills),
            "categorized_candidate_skills": categorize_skills(cand_skills),
            "skill_levels": self.skill_levels(clean),
            "years": years,
            "resume_text": resume_text,
            "feats": feats,
        }
        log.info("Analyzed resume for '%s' | match=%.1f%% level=%s weeks=%.1f compat=%.1f%% (%.2fs)",
                 result["job_title"], result["match_pct"], level, weeks, composite_compat, time.time() - t0)
        return result

    def rank_jobs(self, resume_text, years, top_n=10, domain="All"):
        self._require_valid_resume(resume_text)
        years = extract_resume_metadata(resume_text)["extracted_years"]
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
            matched_s = sorted(js & set(cand))
            missing_s = sorted(js - set(cand))
            transferable = find_transferable_credits(set(cand), set(missing_s))
            rows.append({
                "job_id": idx,
                "title": self.job_titles[idx],
                "domain": self.job_domains[idx],
                "fit_score": round(fit * 100, 1),
                "skill_match_pct": round(ratio * 100, 1),
                "text_match_pct": round(float(sims[idx]) * 100, 1),
                "level": level,
                "weeks_to_ready": round(weeks, 1),
                "matched": matched_s,
                "missing": missing_s,
                "transferable_count": len(transferable),
            })
        log.info("Ranked %d jobs (domain=%s)", len(rows), domain)
        return pd.DataFrame(rows)

    def report_text(self, result):
        lines = [
            "============================================================",
            "  CAREERPATH AI: CAREER TRAJECTORY & SKILL GAP REPORT",
            "============================================================",
            f"Target Role: {result['job_title']} ({result['domain']})",
            f"Overall Compatibility Score: {result.get('composite_compat', result['match_pct']):.1f} / 100",
            f"Direct Skill Match: {result['skill_match_pct']:.1f}%",
            f"Transferable Skill Match: {result.get('transferable_match_pct', result['skill_match_pct']):.1f}%",
            f"Textual Semantic Match: {result['match_pct']:.1f}%",
            f"Assessed Career Readiness Level: {result['level']}",
            f"Estimated Timeline to Ready: {result['weeks']:.1f} weeks (at 10 hours/week)",
            f"Resume Content Strength Score: {result['strength']}/100",
            "",
            "----------------- SKILLS BREAKDOWN -----------------",
            "Matched Core Skills: " + (", ".join(result["matched"]) or "None detected"),
            "Missing Skills: " + (", ".join(result["missing"]) or "None"),
        ]
        if result.get("transferable"):
            lines.append("")
            lines.append("Transferable Skill Bridges:")
            for t in result["transferable"][:4]:
                lines.append(f"  * {t['source_skill']} -> {t['target_skill']} ({t['percentage']}% transferable bridge)")

        lines += ["", "----------------- RESUME FEEDBACK -----------------"]
        lines += [f"- {t}" for t in result["nlp"]["tips"]]
        lines += ["", "----------------- ALTERNATE CAREER PATHS -----------------"]
        for a in result["alternates"]:
            lines.append(f"- {a['domain']} (Fit: {a['fit']:.0f}%) | Best Match: {a['job']} | Level: {a['level']} | Ready: {a['weeks']:.1f} weeks")

        ready = date.today() + timedelta(weeks=result["weeks"])
        lines += [
            "",
            f"Projected Readiness Date: {ready.strftime('%d %B %Y')}",
            "Notice: Guidance generated for academic and career planning purposes.",
            "============================================================"
        ]
        return "\n".join(lines)
