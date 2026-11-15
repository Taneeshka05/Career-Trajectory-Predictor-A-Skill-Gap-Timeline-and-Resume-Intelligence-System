"""
nlp_feedback.py
Advanced NLP Resume Quality Audit, Action Verb Engineering,
and Evidence-Based Resume Feedback Engine.
"""

import re
from typing import Dict, List, Tuple, Any
from core import config as cfg

# Expanded action verbs categorized by impact
STRONG_VERBS = [
    "developed", "designed", "improved", "automated", "reduced", "delivered",
    "optimized", "launched", "implemented", "streamlined", "created", "engineered",
    "architected", "spearheaded", "accelerated", "orchestrated", "refactored",
    "transformed", "maximized", "benchmarked", "deployed", "scaled", "resolved",
    "mentored", "collaborated", "curated", "integrated", "constructed", "executed"
]

# Expanded passive and weak phrases with strong replacements
REPLACEMENTS = {
    "responsible for": "led",
    "worked on": "developed",
    "helped with": "delivered",
    "involved in": "implemented",
    "duties included": "orchestrated",
    "assisted in": "optimized",
    "tasked with": "engineered",
    "handled": "executed",
    "participated in": "collaborated on",
    "took part in": "accelerated",
    "served as": "directed",
    "was part of": "co-developed",
    "tried to": "engineered solutions to",
    "attempted": "researched and executed",
    "dealt with": "resolved",
    "worked with": "partnered with",
    "in charge of": "spearheaded",
}

WEAK_PHRASES = list(REPLACEMENTS.keys())

STRONG_REGEX = re.compile(r"\b(" + "|".join(re.escape(v) for v in STRONG_VERBS) + r")\b", re.IGNORECASE)
WEAK_REGEX = re.compile(r"\b(" + "|".join(re.escape(p) for p in WEAK_PHRASES) + r")\b", re.IGNORECASE)

# Vague / filler phrases that diminish resume impact
VAGUE_WORDS = [
    "various", "multiple things", "etc", "and so on", "day-to-day", "good communication",
    "hard worker", "team player", "punctual", "self-motivated", "fast learner"
]
VAGUE_REGEX = re.compile(r"\b(" + "|".join(re.escape(w) for w in VAGUE_WORDS) + r")\b", re.IGNORECASE)

SECTION_WORDS = {
    "contact": ["email", "phone", "linkedin", "github", "address", "portfolio"],
    "summary": ["summary", "profile", "objective", "about me", "professional summary"],
    "education": ["graduate", "student", "degree", "university", "college", "bca", "mca", "btech", "msc", "bsc"],
    "skills": ["skills", "technical skills", "technologies", "competencies", "tools", "stack"],
    "experience": ["experience", "employment", "work history", "worked", "internship", "professional experience"],
    "projects": ["project", "projects", "capstone", "portfolio", "application", "system"],
    "certifications": ["certifications", "certified", "certificates", "credentials", "licenses"],
}


def split_sentences(text: str) -> List[str]:
    """Splits text into sentences based on punctuation and line breaks."""
    parts = re.split(r"[.!?\n]+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


def analyze_vocabulary(words: List[str]) -> Dict[str, Any]:
    """Evaluates vocabulary richness, repetition, and type-token ratio."""
    if not words:
        return {"ttr": 0.0, "repeated": []}
    low_words = [w.lower() for w in words if len(w) > 3]
    unique_words = set(low_words)
    ttr = len(unique_words) / max(len(low_words), 1)
    
    # Detect high-frequency non-stopword repetitions
    freq: Dict[str, int] = {}
    for w in low_words:
        freq[w] = freq.get(w, 0) + 1
    repeated = [w for w, count in freq.items() if count >= 4 and w not in cfg.STRONG_VERBS]
    return {"ttr": round(ttr, 2), "repeated": repeated[:5]}


def generate_star_bullet_suggestions(cand_skills: List[str], domain: str = "Technology") -> List[Dict[str, str]]:
    """
    Generates contextual STAR (Situation-Task-Action-Result) bullet framework templates
    tailored to the user's detected skills, without fabricating experiences.
    """
    templates = [
        ("Automated {skill} data processing pipelines, reducing latency by 35% and saving 8 team-hours weekly.", "Optimization"),
        ("Architected and deployed a production-grade {skill} solution, serving 1,000+ active user requests daily with 99.9% uptime.", "Architecture"),
        ("Refactored legacy workflows using {skill}, improving query execution speed by 40% and cutting infrastructure overhead.", "Performance"),
        ("Implemented automated test suites and validation scripts in {skill}, elevating test coverage to 92% across all modules.", "Quality Assurance"),
        ("Designed an end-to-end {skill} feature pipeline, cutting client report delivery turnaround from 3 days to under 4 hours.", "Delivery"),
    ]
    
    results = []
    sample_skills = cand_skills[:3] if cand_skills else ["Python", "SQL"]
    for i, (tmpl, category) in enumerate(templates):
        chosen_skill = sample_skills[i % len(sample_skills)].title()
        bullet = tmpl.format(skill=chosen_skill)
        results.append({
            "category": category,
            "skill": chosen_skill,
            "template": bullet,
            "formula": "Action Verb + Technical Scope ({skill}) + Quantifiable Metric + Business Outcome"
        })
    return results


def analyze_text(text: str, required_skills: List[str] = None, found_skills: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive linguistic, structural, and impact analysis of resume text.
    """
    sentences = split_sentences(text)
    words = re.findall(r"[A-Za-z0-9']+", text)
    strong = [m.lower() for m in STRONG_REGEX.findall(text)]
    weak = [m.lower() for m in WEAK_REGEX.findall(text)]
    vague_matches = [m.lower() for m in VAGUE_REGEX.findall(text)]
    
    # Metrics and quantitative indicators
    metric_matches = re.findall(
        r"\b\d+(?:[.,]\d+)*\s*%|\b\d+(?:[.,]\d+)*\+?\s*(?:x|k|m|lakh|crore|users?|clients?|requests?|queries|ms|seconds?|minutes?|hours?|days?|percent)\b|\$\s*\d+(?:[.,]\d+)*|\b\d+\+\s*(?:years?|projects?|features?|tools?)",
        text.lower()
    )
    quantified = len(metric_matches)
    
    low = text.lower()
    sections = {name: any(w in low for w in ws) for name, ws in SECTION_WORDS.items()}
    
    coverage = 0.0
    if required_skills:
        coverage = len(set(required_skills) & set(found_skills or [])) / max(len(set(required_skills)), 1)
        
    avg_len = len(words) / len(sentences) if sentences else 0.0
    action_ratio = len(strong) / max(len(strong) + len(weak), 1)
    vocab = analyze_vocabulary(words)
    
    # Generate targeted, prioritized advice
    tips = []
    for w in sorted(set(weak)):
        replacement = REPLACEMENTS.get(w, "developed")
        tips.append(f"Replace passive phrase '{w}' with high-impact active verb like '{replacement}'.")
    
    for v in sorted(set(vague_matches))[:3]:
        tips.append(f"Replace vague filler phrase '{v}' with specific technical deliverables or metrics.")
        
    if quantified == 0:
        tips.append("Add measurable outcomes (e.g. 'reduced processing time by 30%', 'scaled to 5,000+ users').")
    elif quantified < 3:
        tips.append("Strengthen project bullets with more quantifiable metrics and measurable business outcomes.")
        
    if len(words) < 80:
        tips.append("Resume length is minimal. Expand project descriptions, architectural choices, and tools used.")
    if avg_len > 24:
        tips.append("Some sentences are overly verbose. Aim for crisp bullet points between 12 and 20 words.")
        
    missing_sections = [s.capitalize() for s, present in sections.items() if not present and s not in ["certifications"]]
    if missing_sections:
        tips.append(f"Missing distinct sections: {', '.join(missing_sections)}. Add clear section headers.")
        
    if vocab["repeated"]:
        tips.append(f"Frequent word repetitions detected: {', '.join(vocab['repeated'])}. Use varied technical vocabulary.")
        
    if not tips:
        tips.append("Language is strong and action-oriented. Continue updating metrics as project milestones grow.")

    # Calculate rubric breakdown
    metric_score = min(quantified * 25, 100)
    action_score = min(len(strong) * 15, 100) if not weak else max(0, min(len(strong) * 15, 100) - len(weak) * 10)
    structure_score = int(sum(1 for v in sections.values() if v) / len(sections) * 100)
    skill_density_score = min(int(coverage * 100) if required_skills else min(len(found_skills or []) * 12, 100), 100)

    rubric = {
        "Measurable Impact": metric_score,
        "Action Verbs": action_score,
        "Section Completeness": structure_score,
        "Skill Density": skill_density_score,
    }

    return {
        "n_sentences": len(sentences),
        "n_words": len(words),
        "avg_sentence_len": round(avg_len, 1),
        "strong": sorted(set(strong)),
        "weak": sorted(set(weak)),
        "n_strong": len(strong),
        "n_weak": len(weak),
        "vague": sorted(set(vague_matches)),
        "action_ratio": round(action_ratio, 2),
        "quantified": quantified,
        "metric_samples": metric_matches[:5],
        "sections": sections,
        "keyword_coverage": round(coverage, 2),
        "vocabulary": vocab,
        "rubric": rubric,
        "tips": tips,
    }


def strength_score(nlp: Dict[str, Any], lang_prob: float) -> float:
    """
    Transparent, explainable resume strength score (0-100).
    Rubric:
    - 30%: Measurable impact & quantitative achievements
    - 25%: Action-oriented language ratio & strong verbs
    - 25%: Technical keyword / skill coverage
    - 20%: Section structure completeness & Neural language score
    """
    rubric = nlp.get("rubric", {})
    metric_comp = min(nlp["quantified"] / 3.0, 1.0) * 30.0
    action_comp = (nlp["action_ratio"] * 0.7 + min(nlp["n_strong"] / 5.0, 1.0) * 0.3) * 25.0
    skill_comp = nlp["keyword_coverage"] * 25.0
    
    struct_ratio = sum(1 for v in nlp["sections"].values() if v) / max(len(nlp["sections"]), 1)
    quality_comp = (struct_ratio * 0.5 + lang_prob * 0.5) * 20.0
    
    total = metric_comp + action_comp + skill_comp + quality_comp
    return round(float(np_clip(total, 5.0, 100.0)), 1)


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def highlight_html(text: str) -> str:
    """Highlights weak phrases (red) and strong action verbs (green) in HTML."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = WEAK_REGEX.sub(lambda m: f"<span class='hl-weak' title='Passive/Weak phrase'>{m.group(0)}</span>", safe)
    safe = STRONG_REGEX.sub(lambda m: f"<span class='hl-strong' title='Strong action verb'>{m.group(0)}</span>", safe)
    return safe.replace("\n", "<br>")


def auto_improve(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Replaces weak phrases with professional active verbs while preserving case.
    Returns improved text and list of change mappings.
    """
    changes = []

    def swap(match):
        original = match.group(0)
        low = original.lower()
        new = REPLACEMENTS.get(low, "developed")
        if original.isupper():
            new = new.upper()
        elif original[0].isupper():
            new = new.capitalize()
        changes.append({"before": original, "after": new})
        return new

    improved = WEAK_REGEX.sub(swap, text)
    return improved, changes
