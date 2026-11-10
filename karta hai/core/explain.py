import pandas as pd
from core import config as cfg

READABLE = {
    "missing_n": "number of missing skills", "missing_difficulty": "difficulty of missing skills",
    "years_exp": "years of experience", "match_score": "text match with job",
    "skill_match_ratio": "share of required skills you have", "n_resume_skills": "skills on resume",
    "matched_n": "number of matched skills", "avg_diff_matched": "difficulty of skills you have",
}


def weeks_contributions(engine, feats):
    reg = engine.b["reg"]
    row = pd.DataFrame([feats])[cfg.REG_FEATURES]
    z = reg["scaler"].transform(row)[0]
    model = reg["best"]
    df = pd.DataFrame({"factor": [READABLE[f] for f in cfg.REG_FEATURES],
                       "value": row.iloc[0].to_numpy(), "contribution": model.coef_ * z})
    df = df.sort_values("contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return float(model.intercept_), df


def level_contributions(engine, feats, level):
    clf = engine.b["clf"]
    lr = clf["models"]["Logistic Regression"]
    row = pd.DataFrame([feats])[cfg.CLF_FEATURES]
    z = clf["scaler"].transform(row)[0]
    idx = list(lr.classes_).index(level)
    df = pd.DataFrame({"factor": [READABLE[f] for f in cfg.CLF_FEATURES],
                       "value": row.iloc[0].to_numpy(), "contribution": lr.coef_[idx] * z})
    return df.sort_values("contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def sentences(df, unit, top=3):
    out = []
    for r in df.head(top).itertuples():
        verb = "adds" if r.contribution > 0 else "reduces"
        out.append(f"{r.factor.capitalize()} {verb} about {abs(r.contribution):.1f} {unit}.")
    return out