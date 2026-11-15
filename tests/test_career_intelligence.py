import pytest
from core import config as cfg
from core.engine import CareerEngine


@pytest.fixture(scope="module")
def engine():
    return CareerEngine.load_or_train()


def test_analyze_with_transferable_bridges(engine):
    # Resume with PyTorch instead of TensorFlow for an ML Engineer job
    resume = (
        "MCA graduate with 3 years of experience. Machine learning engineer skilled in python, "
        "pytorch, scikit learn, pandas, numpy. Designed deep learning models."
    )
    result = engine.analyze(resume, 3, 120)  # ML Engineer job
    assert "composite_compat" in result
    assert 0 <= result["composite_compat"] <= 100
    assert "transferable" in result
    assert "gap_explanations" in result
    assert isinstance(result["gap_explanations"], list)
    if result["missing"]:
        assert len(result["gap_explanations"]) == len(result["missing"])


def test_alternate_careers_rich_fields(engine):
    resume = "BCA graduate with 2 years experience in java, sql, and html."
    alts = engine.alternate_careers("java sql html", ["java", "sql", "html"], 2)
    assert len(alts) >= 3
    for a in alts:
        assert "domain" in a
        assert "fit" in a
        assert "rationale" in a
        assert "overlap" in a
        assert "missing" in a


def test_screen_with_transferable_credits(engine):
    resume = (
        "Professional Experience\n"
        "Machine Learning Engineer, 2021 - Present\n"
        "Built Python and PyTorch models with scikit learn and pandas.\n"
        "Education\n"
        "MCA, Example University"
    )
    scr = engine.screen(resume, 4, 120)
    assert "fit_score" in scr
    assert "transferable" in scr
    assert scr["years"] == engine.parse_years(resume)
    assert 0 <= scr["fit_score"] <= 100
