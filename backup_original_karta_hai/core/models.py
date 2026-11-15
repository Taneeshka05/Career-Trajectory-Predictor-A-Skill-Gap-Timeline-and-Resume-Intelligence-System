import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

from sklearn.base import clone
from sklearn.inspection import permutation_importance 
from sklearn.pipeline import make_pipeline
from core import config as cfg
from core.data_generator import generate_datasets
from core.preprocess import prepare_text, fit_tfidf, build_pairs
from core.logger_setup import get_logger

log = get_logger("models")

BUNDLE_PATH = cfg.ARTIFACT_DIR / "bundle.joblib"
NN_PATH = cfg.ARTIFACT_DIR / "language_nn.keras"
LEVEL_ORDER = ["Junior", "Mid", "Senior"]


def split_pairs(pairs):
    train_idx, test_idx = train_test_split(np.arange(len(pairs)), test_size=0.2,
                                           random_state=cfg.SEED, stratify=pairs["level"])
    return pairs.iloc[train_idx], pairs.iloc[test_idx]


def train_regression(train_df, test_df):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[cfg.REG_FEATURES])
    X_test = scaler.transform(test_df[cfg.REG_FEATURES])
    y_train = train_df["time_to_ready"].to_numpy()
    y_test = test_df["time_to_ready"].to_numpy()

    models = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=1.0)}
    rows = []
    preds = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds[name] = p
        mse = mean_squared_error(y_test, p)
        rows.append({"Model": name, "MSE": mse, "RMSE": mse ** 0.5,
                     "MAE": mean_absolute_error(y_test, p), "R2": r2_score(y_test, p)})
        log.info("Regression %s | MSE=%.3f R2=%.3f", name, mse, r2_score(y_test, p))
    results = pd.DataFrame(rows)
    best_name = results.sort_values("R2", ascending=False).iloc[0]["Model"]
    return {"models": models, "scaler": scaler, "results": results, "best_name": best_name,
            "best": models[best_name], "y_test": y_test, "preds": preds}


def train_classifiers(train_df, test_df):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[cfg.CLF_FEATURES])
    X_test = scaler.transform(test_df[cfg.CLF_FEATURES])
    y_train = train_df["level"].to_numpy()
    y_test = test_df["level"].to_numpy()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=cfg.SEED),
        "SVM": SVC(kernel="rbf", random_state=cfg.SEED),
    }
    rows = []
    preds = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds[name] = p
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, p),
            "Precision": precision_score(y_test, p, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, p, average="weighted", zero_division=0),
            "F1": f1_score(y_test, p, average="weighted", zero_division=0),
        })
        log.info("Classifier %s | accuracy=%.3f", name, rows[-1]["Accuracy"])
    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    cm = confusion_matrix(y_test, preds[best_name], labels=LEVEL_ORDER)
    return {"models": models, "scaler": scaler, "results": results, "best_name": best_name,
            "best": models[best_name], "cm": cm}


def build_skill_matrix(resumes):
    X = np.zeros((len(resumes), len(cfg.ALL_SKILLS)))
    for i, sk in enumerate(resumes["skills"]):
        for s in sk:
            X[i, cfg.ALL_SKILLS.index(s)] = 1
    return X


def run_pca_kmeans(resumes, X_skill):
    pca = PCA(n_components=2, random_state=cfg.SEED)
    coords = pca.fit_transform(X_skill)
    kmeans = KMeans(n_clusters=len(cfg.DOMAIN_NAMES), n_init=10, random_state=cfg.SEED)
    labels = kmeans.fit_predict(X_skill)
    centroids = np.array([X_skill[(resumes["domain"] == d).to_numpy()].mean(axis=0)
                          for d in cfg.DOMAIN_NAMES])
    map_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                           "domain": resumes["domain"].to_numpy(), "cluster": labels})
    log.info("PCA variance explained: %s", np.round(pca.explained_variance_ratio_, 3))
    return {"pca": pca, "map_df": map_df, "centroids": centroids,
            "crosstab": pd.crosstab(resumes["domain"], labels)}


def train_language_nn(resumes):
    train_idx, test_idx = train_test_split(np.arange(len(resumes)), test_size=0.2,
                                           random_state=cfg.SEED,
                                           stratify=resumes["language_label"])
    vec = TfidfVectorizer(max_features=300)
    X_train = vec.fit_transform(resumes["clean"].iloc[train_idx]).toarray().astype("float32")
    X_test = vec.transform(resumes["clean"].iloc[test_idx]).toarray().astype("float32")
    y_train = resumes["language_label"].iloc[train_idx].to_numpy()
    y_test = resumes["language_label"].iloc[test_idx].to_numpy()

    nn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = nn.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=64,
                     callbacks=[stop], verbose=0)
    nn_pred = (nn.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)

    base = LogisticRegression(max_iter=1000)
    base.fit(X_train, y_train)
    base_pred = base.predict(X_test)

    comparison = pd.DataFrame({
        "Model": ["Neural Network", "Logistic Regression"],
        "Accuracy": [accuracy_score(y_test, nn_pred), accuracy_score(y_test, base_pred)],
        "F1": [f1_score(y_test, nn_pred), f1_score(y_test, base_pred)],
    })
    log.info("Neural network | accuracy=%.3f epochs=%d", comparison.loc[0, "Accuracy"],
             len(history.history["loss"]))
    return {"nn": nn, "vec": vec, "history": history.history, "comparison": comparison,
            "cm": confusion_matrix(y_test, nn_pred)}

def reliability_report(pairs, train_df, test_df, clf):
    X_all = pairs[cfg.CLF_FEATURES]
    y_all = pairs["level"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    rows = []
    for name, model in clf["models"].items():
        pipe = make_pipeline(StandardScaler(), clone(model))
        scores = cross_val_score(pipe, X_all, y_all, cv=skf, scoring="accuracy")
        rows.append({"Model": name, "CV mean accuracy": scores.mean(), "CV std": scores.std()})
    cv = pd.DataFrame(rows)

    pipe = make_pipeline(StandardScaler(), clone(clf["best"]))
    sizes, _, test_scores = learning_curve(pipe, X_all, y_all, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
                                           cv=3, scoring="accuracy", shuffle=True,
                                           random_state=cfg.SEED)
    curve = pd.DataFrame({"training samples": sizes, "validation accuracy": test_scores.mean(axis=1)})

    rng = np.random.default_rng(cfg.SEED)
    X_train = clf["scaler"].transform(train_df[cfg.CLF_FEATURES])
    X_test = clf["scaler"].transform(test_df[cfg.CLF_FEATURES])
    y_train = train_df["level"].to_numpy()
    y_test = test_df["level"].to_numpy()
    noise_rows = []
    for p in [0.0, 0.1, 0.2, 0.3]:
        noisy = y_train.copy()
        flip = rng.choice(len(noisy), size=int(p * len(noisy)), replace=False)
        for i in flip:
            others = [l for l in LEVEL_ORDER if l != noisy[i]]
            noisy[i] = others[int(rng.integers(0, 2))]
        m = LogisticRegression(max_iter=1000).fit(X_train, noisy)
        noise_rows.append({"label noise %": int(p * 100), "test accuracy": accuracy_score(y_test, m.predict(X_test))})
    noise = pd.DataFrame(noise_rows)

    perm = permutation_importance(clf["best"], X_test, y_test, n_repeats=5, random_state=cfg.SEED,
                                  scoring="accuracy")
    importance = pd.DataFrame({"feature": cfg.CLF_FEATURES, "importance": perm.importances_mean})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)

    by_source = pd.DataFrame({"source": test_df["source"].to_numpy(),
                              "correct": clf["best"].predict(X_test) == y_test})
    by_source = by_source.groupby("source")["correct"].agg(["mean", "count"]).reset_index()
    by_source.columns = ["data source", "accuracy", "test samples"]
    log.info("Reliability report ready | CV accuracy range %.3f-%.3f", cv["CV mean accuracy"].min(),
             cv["CV mean accuracy"].max())
    return {"cv": cv, "curve": curve, "noise": noise, "importance": importance, "by_source": by_source}


def train_all(resume_csv=None, only_real=False):
    start = time.time()
    rng = np.random.default_rng(cfg.SEED)
    tf.random.set_seed(cfg.SEED)
    log.info("Training pipeline started")

    resumes, jobs, _ = generate_datasets(rng, resume_csv, only_real)
    resumes, jobs = prepare_text(resumes, jobs)
    tfidf = fit_tfidf(resumes, jobs)
    pairs, _ = build_pairs(resumes, jobs, tfidf, rng)
    pairs["source"] = resumes["source"].to_numpy()
    train_df, test_df = split_pairs(pairs)

    reg = train_regression(train_df, test_df)
    clf = train_classifiers(train_df, test_df)
    pca_info = run_pca_kmeans(resumes, build_skill_matrix(resumes))
    nn_info = train_language_nn(resumes)
    reliability = reliability_report(pairs, train_df, test_df, clf)

    sample = np.random.default_rng(1).choice(len(reg["y_test"]), size=600, replace=False)
    reg_scatter = pd.DataFrame({"actual": reg["y_test"][sample],
                                "predicted": reg["preds"][reg["best_name"]][sample]})
    scaler_table = pd.DataFrame({
        "feature": cfg.REG_FEATURES + cfg.CLF_FEATURES,
        "mean": list(reg["scaler"].mean_) + list(clf["scaler"].mean_),
        "std": list(reg["scaler"].scale_) + list(clf["scaler"].scale_),
        "used_for": ["regression"] * len(cfg.REG_FEATURES) + ["classification"] * len(cfg.CLF_FEATURES),
    })
    stats = {
        "n_resumes": len(resumes), "n_jobs": len(jobs), "n_skills": len(cfg.ALL_SKILLS),
        "level_counts": pairs["level"].value_counts().to_dict(),
        "domain_counts": resumes["domain"].value_counts().to_dict(),
        "source_counts": resumes["source"].value_counts().to_dict(),
        "sample_resumes": resumes[["domain", "years_exp", "resume_text"]].head(8),
    }

    bundle = {
        "jobs": jobs, "tfidf": tfidf, "reg": reg, "clf": clf, "pca": pca_info,
        "nn_vec": nn_info["vec"], "nn_history": nn_info["history"],
        "nn_comparison": nn_info["comparison"], "nn_cm": nn_info["cm"],
        "reg_scatter": reg_scatter, "scaler_table": scaler_table, "stats": stats,
        "reliability": reliability,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    nn_info["nn"].save(NN_PATH)
    log.info("Training finished in %.1f seconds, artifacts saved", time.time() - start)
    return bundle


def load_artifacts():
    bundle = joblib.load(BUNDLE_PATH)
    nn = tf.keras.models.load_model(NN_PATH)
    return bundle, nn


def artifacts_exist():
    return BUNDLE_PATH.exists() and NN_PATH.exists()