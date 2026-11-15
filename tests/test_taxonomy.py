import pytest
from core.taxonomy import (
    SKILL_TAXONOMY,
    SOFT_SKILLS,
    SKILL_ALIASES,
    TRANSFERABLE_SKILLS,
    get_skill_category,
    find_transferable_credits,
    categorize_skills,
)
from core.preprocess import extract_skills, extract_soft_skills, count_skill_mentions


def test_taxonomy_categories_populated():
    assert len(SKILL_TAXONOMY) >= 6
    assert "Programming Languages" in SKILL_TAXONOMY
    assert "Machine Learning & AI" in SKILL_TAXONOMY
    assert "python" in SKILL_TAXONOMY["Programming Languages"]


def test_alias_mapping_in_extract_skills():
    # 'js' should resolve to 'javascript', 'postgres' to 'sql', 'k8s' to 'kubernetes'
    clean = "proficient in js and postgres with experience deploying on k8s"
    skills = extract_skills(clean)
    assert "javascript" in skills
    assert "sql" in skills
    assert "kubernetes" in skills


def test_soft_skills_extraction():
    clean = "strong communication problem solving and agile experience"
    soft = extract_soft_skills(clean)
    assert "communication" in soft
    assert "problem solving" in soft
    assert "agile" in soft


def test_transferable_credits_calculation():
    # If candidate has pytorch and job requires tensorflow
    owned = {"pytorch"}
    missing = {"tensorflow"}
    credits = find_transferable_credits(owned, missing)
    assert len(credits) == 1
    assert credits[0]["target_skill"] == "tensorflow"
    assert credits[0]["source_skill"] == "pytorch"
    assert credits[0]["percentage"] == 85


def test_categorize_skills_helper():
    sample = ["python", "docker", "selenium", "react"]
    categorized = categorize_skills(sample)
    assert "Programming Languages" in categorized
    assert "Cloud & DevOps" in categorized
    assert "python" in categorized["Programming Languages"]
    assert "docker" in categorized["Cloud & DevOps"]


def test_skill_mentions_counting_with_aliases():
    clean = "python developer experienced with python and py scripts"
    counts = count_skill_mentions(clean)
    assert "python" in counts
    # 'python' mentioned twice, 'py' mentioned once -> total 3
    assert counts["python"] >= 2
