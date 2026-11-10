import re
from core import config as cfg

STRONG_REGEX = re.compile(r"\b(" + "|".join(cfg.STRONG_VERBS) + r")\b", re.IGNORECASE)
WEAK_REGEX = re.compile(r"\b(" + "|".join(cfg.WEAK_PHRASES) + r")\b", re.IGNORECASE)
SECTION_WORDS = {
    "education": ["graduate", "student", "degree", "university", "bca", "mca", "btech", "msc", "bsc"],
    "skills": ["skills", "technical skills", "tools", "technologies"],
    "experience": ["experience", "worked", "developed", "created", "delivered", "implemented"],
    "projects": ["project", "projects", "dashboard", "dashboards", "capstone"],
}


def split_sentences(text):
    parts = re.split(r"[.!?\n]+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


def analyze_text(text, required_skills=None, found_skills=None):
    sentences = split_sentences(text)
    words = re.findall(r"[A-Za-z0-9']+", text)
    strong = [m.lower() for m in STRONG_REGEX.findall(text)]
    weak = [m.lower() for m in WEAK_REGEX.findall(text)]
    quantified = len(re.findall(r"\d+\s*%|\d+\s*(?:x|k|lakh|users|clients|projects)\b", text.lower()))
    low = text.lower()
    sections = {name: any(w in low for w in ws) for name, ws in SECTION_WORDS.items()}

    coverage = 0.0
    if required_skills:
        coverage = len(set(required_skills) & set(found_skills or [])) / len(set(required_skills))

    avg_len = len(words) / len(sentences) if sentences else 0.0
    action_ratio = len(strong) / max(len(strong) + len(weak), 1)

    tips = []
    for w in sorted(set(weak)):
        tips.append(f"Replace '{w}' with a stronger verb such as '{cfg.REPLACEMENTS[w]}'.")
    if quantified == 0:
        tips.append("Add measurable results, for example 'reduced report time by 30%'.")
    if len(words) < 60:
        tips.append("Resume is very short. Add projects, tools used and outcomes.")
    if avg_len > 22:
        tips.append("Sentences are long. Keep bullet points under 20 words.")
    for name, present in sections.items():
        if not present:
            tips.append(f"No clear '{name}' section found. Add one.")
    if not tips:
        tips.append("Language is strong. Keep adding measurable impact for each project.")

    return {
        "n_sentences": len(sentences), "n_words": len(words), "avg_sentence_len": round(avg_len, 1),
        "strong": sorted(set(strong)), "weak": sorted(set(weak)), "n_strong": len(strong),
        "n_weak": len(weak), "action_ratio": action_ratio, "quantified": quantified,
        "sections": sections, "keyword_coverage": coverage, "tips": tips,
    }


def strength_score(nlp, lang_prob):
    score = (0.45 * lang_prob + 0.20 * min(nlp["quantified"] / 3, 1)
             + 0.20 * nlp["keyword_coverage"] + 0.15 * min(nlp["n_strong"] / 5, 1))
    return round(100 * score, 1)


def highlight_html(text):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = WEAK_REGEX.sub(lambda m: f"<span class='hl-weak'>{m.group(0)}</span>", safe)
    safe = STRONG_REGEX.sub(lambda m: f"<span class='hl-strong'>{m.group(0)}</span>", safe)
    return safe.replace("\n", "<br>")


def auto_improve(text):
    changes = []

    def swap(match):
        original = match.group(0)
        new = cfg.REPLACEMENTS[original.lower()]
        if original[0].isupper():
            new = new.capitalize()
        changes.append({"before": original, "after": new})
        return new

    improved = WEAK_REGEX.sub(swap, text)
    return improved, changes