import numpy as np
import pandas as pd
import pytest
from core.ml_pipeline import DataValidator, MLWorkflowEngine


@pytest.fixture
def sample_regression_df():
    rng = np.random.default_rng(123)
    n = 150
    x1 = rng.integers(1, 10, size=n)
    x2 = rng.uniform(0.1, 0.9, size=n)
    cat = rng.choice(["DevOps", "DataScience", "WebDev"], size=n)
    # Target: y = 2*x1 - 3*x2 + noise
    y = np.round(2.5 * x1 - 3.0 * x2 + 5.0 + rng.normal(0, 0.5, size=n), 2)
    return pd.DataFrame({"skill_count": x1, "similarity": x2, "domain": cat, "weeks": y})


@pytest.fixture
def sample_classification_df():
    rng = np.random.default_rng(456)
    n = 180
    exp = rng.integers(0, 12, size=n)
    score = rng.uniform(0.2, 0.9, size=n)
    level = ["Junior" if e < 3 else ("Mid" if e < 7 else "Senior") for e in exp]
    return pd.DataFrame({"years": exp, "fit": score, "tier": level})


def test_data_validator_audit(sample_regression_df):
    audit = DataValidator.audit(sample_regression_df)
    assert audit["valid"] is True
    assert audit["rows"] == 150
    assert audit["cols"] == 4
    assert audit["missing_total"] == 0
    assert audit["health_score"] >= 80


def test_data_validator_rejects_empty():
    empty_df = pd.DataFrame()
    audit = DataValidator.audit(empty_df)
    assert audit["valid"] is False


def test_ml_pipeline_regression(sample_regression_df):
    engine = MLWorkflowEngine(
        sample_regression_df,
        target_col="weeks",
        feature_cols=["skill_count", "similarity", "domain"],
        task_type="Regression"
    )
    result = engine.run_pipeline(test_size=0.2, random_state=42)
    assert result["task_type"] == "Regression"
    assert engine.evaluation_results is not None
    assert len(engine.evaluation_results) == 3  # Linear, Ridge, Random Forest
    assert engine.best_model_name in ["Linear Regression", "Ridge Regression", "Random Forest Regressor"]

    # Test live prediction on new unseen sample
    input_sample = {"skill_count": 5, "similarity": 0.6, "domain": "DataScience"}
    pred = engine.predict_new(input_sample)
    assert isinstance(pred, (float, np.floating, int, np.integer))
    assert 5.0 <= pred <= 25.0


def test_ml_pipeline_classification(sample_classification_df):
    engine = MLWorkflowEngine(
        sample_classification_df,
        target_col="tier",
        feature_cols=["years", "fit"],
        task_type="Classification"
    )
    result = engine.run_pipeline(test_size=0.2, random_state=42)
    assert result["task_type"] == "Classification"
    assert engine.evaluation_results is not None
    assert "Accuracy" in engine.evaluation_results.columns
    assert "F1-Score" in engine.evaluation_results.columns

    # Live prediction
    pred = engine.predict_new({"years": 1, "fit": 0.3})
    assert pred in ["Junior", "Mid", "Senior"]
