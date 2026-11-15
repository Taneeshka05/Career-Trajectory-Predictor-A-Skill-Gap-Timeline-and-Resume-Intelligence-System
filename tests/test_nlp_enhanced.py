import pytest
from core import nlp_feedback as nlp


def test_auto_improve_expanded_phrases():
    text = "Tasked with reporting. Dealt with server outages. In charge of database queries."
    improved, changes = nlp.auto_improve(text)
    assert "engineered" in improved.lower()
    assert "resolved" in improved.lower()
    assert "spearheaded" in improved.lower()
    assert len(changes) == 3


def test_vocabulary_analysis():
    words = ["python", "developed", "dashboard", "python", "python", "python", "scaled", "cloud"]
    vocab = nlp.analyze_vocabulary(words)
    assert 0.0 < vocab["ttr"] <= 1.0
    assert "python" in vocab["repeated"]


def test_star_bullet_generator():
    skills = ["docker", "python"]
    bullets = nlp.generate_star_bullet_suggestions(skills)
    assert len(bullets) >= 4
    for b in bullets:
        assert "template" in b
        assert "formula" in b
        assert "Docker" in b["template"] or "Python" in b["template"]


def test_analyze_text_rubric_and_metrics():
    text = (
        "Professional Summary: Senior Engineer.\n"
        "Technical Skills: Python, Docker, SQL.\n"
        "Experience: Architected data pipelines using Python, reducing processing latency by 35%.\n"
        "Automated deployment workflows using Docker across 5,000+ client requests."
    )
    analysis = nlp.analyze_text(text, required_skills=["python", "docker"], found_skills=["python", "docker"])
    assert analysis["n_strong"] >= 2
    assert analysis["quantified"] >= 2
    assert "Measurable Impact" in analysis["rubric"]
    assert "Action Verbs" in analysis["rubric"]
    score = nlp.strength_score(analysis, lang_prob=0.85)
    assert 0 <= score <= 100
