"""
ml_pipeline.py
End-to-End Dynamic Machine Learning Pipeline Engine.
Implements:
DATA INPUT -> VALIDATION -> EDA -> PREPROCESSING -> TRAINING ->
TESTING ON UNSEEN DATA -> EVALUATION -> FEATURE IMPORTANCE -> LIVE PREDICTION
"""

import io
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from core.logger_setup import get_logger

log = get_logger("ml_pipeline")


class DataValidator:
    """Validates structure, types, missing rates, and quality of tabular datasets."""
    
    @staticmethod
    def load_dataset(file_obj, filename: str) -> pd.DataFrame:
        fn = filename.lower()
        if fn.endswith(".csv"):
            return pd.read_csv(file_obj)
        elif fn.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_obj)
        else:
            raise ValueError(f"Unsupported file format '{filename}'. Please upload a .csv or .xlsx file.")

    @staticmethod
    def audit(df: pd.DataFrame) -> Dict[str, Any]:
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return {"valid": False, "error": "The uploaded dataset contains 0 rows."}
        if n_cols < 2:
            return {"valid": False, "error": "The dataset must contain at least 2 columns (features and a target)."}

        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        total_cells = n_rows * n_cols
        missing_rate = total_missing / max(total_cells, 1)

        duplicates = int(df.duplicated().sum())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Quality health score (0-100)
        health = 100.0
        health -= min(missing_rate * 80.0, 40.0)
        health -= min((duplicates / max(n_rows, 1)) * 40.0, 20.0)
        if n_rows < 30:
            health -= 25.0
        health = max(10.0, round(health, 1))

        return {
            "valid": True,
            "rows": n_rows,
            "cols": n_cols,
            "missing_total": total_missing,
            "missing_rate_pct": round(missing_rate * 100, 2),
            "missing_by_col": missing_counts[missing_counts > 0].to_dict(),
            "duplicates": duplicates,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "health_score": health,
        }


class MLWorkflowEngine:
    """
    Executes real scikit-learn training, testing, comparison, and evaluation
    on custom datasets without data leakage.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, feature_cols: List[str], task_type: str = "Auto"):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        # Determine task type
        if task_type == "Auto":
            target_series = self.df[target_col].dropna()
            if np.issubdtype(target_series.dtype, np.number) and target_series.nunique() > 10:
                self.task_type = "Regression"
            else:
                self.task_type = "Classification"
        else:
            self.task_type = task_type

        self.models: Dict[str, Any] = {}
        self.evaluation_results: Optional[pd.DataFrame] = None
        self.best_model_name: Optional[str] = None
        self.best_pipeline: Optional[Any] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[np.ndarray] = None
        self.test_predictions: Dict[str, np.ndarray] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        self.feature_names_out: List[str] = []
        self.importances: Optional[pd.DataFrame] = None

    def run_pipeline(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Executes complete training and validation cycle:
        1. Feature/target splitting
        2. Train/Test split (stratified if classification)
        3. Preprocessing pipeline (fit on train only)
        4. Model training across candidates
        5. Unseen test evaluation
        """
        # Drop rows where target is missing
        clean_df = self.df.dropna(subset=[self.target_col]).copy()
        X = clean_df[self.feature_cols]
        y = clean_df[self.target_col]

        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Stratify classification if classes have at least 2 instances
        stratify = None
        if self.task_type == "Classification":
            val_counts = y.value_counts()
            if (val_counts >= 2).all():
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        self.X_test = X_test
        self.y_test = y_test.to_numpy()

        # Build column transformer
        transformers = []
        if num_features:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipe, num_features))
        if cat_features:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, cat_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)

        # Extract transformed feature names
        names = []
        if num_features:
            names.extend(num_features)
        if cat_features:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_encoded_names = cat_encoder.get_feature_names_out(cat_features).tolist()
            names.extend(cat_encoded_names)
        self.feature_names_out = names

        if self.task_type == "Regression":
            models_to_train = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=random_state)
            }
            rows = []
            for name, model in models_to_train.items():
                model.fit(X_train_trans, y_train)
                preds = model.predict(X_test_trans)
                self.models[name] = model
                self.test_predictions[name] = preds

                mse = mean_squared_error(self.y_test, preds)
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(self.y_test, preds))
                r2 = float(r2_score(self.y_test, preds))

                rows.append({
                    "Model": name,
                    "R² Score": round(r2, 4),
                    "RMSE": round(rmse, 3),
                    "MAE": round(mae, 3),
                })
            results_df = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
            self.best_model_name = results_df.iloc[0]["Model"]

        else:  # Classification
            models_to_train = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=random_state),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train) - 1)))
            }
            rows = []
            classes = sorted(y.unique())
            for name, model in models_to_train.items():
                model.fit(X_train_trans, y_train)
                preds = model.predict(X_test_trans)
                self.models[name] = model
                self.test_predictions[name] = preds
                self.confusion_matrices[name] = confusion_matrix(self.y_test, preds, labels=classes)

                acc = float(accuracy_score(self.y_test, preds))
                prec = float(precision_score(self.y_test, preds, average="weighted", zero_division=0))
                rec = float(recall_score(self.y_test, preds, average="weighted", zero_division=0))
                f1 = float(f1_score(self.y_test, preds, average="weighted", zero_division=0))

                rows.append({
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "F1-Score": round(f1, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                })
            results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
            self.best_model_name = results_df.iloc[0]["Model"]

        self.evaluation_results = results_df

        # Calculate feature importance for best model
        best_mod = self.models[self.best_model_name]
        importances_list = None
        if hasattr(best_mod, "feature_importances_"):
            importances_list = best_mod.feature_importances_
        elif hasattr(best_mod, "coef_"):
            coef = best_mod.coef_
            importances_list = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances_list is not None and len(importances_list) == len(self.feature_names_out):
            imp_df = pd.DataFrame({
                "feature": self.feature_names_out,
                "importance": importances_list
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            self.importances = imp_df

        log.info("ML Workflow completed | Task: %s | Best Model: %s", self.task_type, self.best_model_name)
        return {
            "task_type": self.task_type,
            "results": results_df,
            "best_model_name": self.best_model_name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def predict_new(self, input_dict: Dict[str, Any]) -> Any:
        """Processes raw user input dictionary and returns live prediction from the best model."""
        if not self.models or not self.best_model_name:
            raise RuntimeError("Pipeline has not been trained yet.")
        row_df = pd.DataFrame([input_dict])[self.feature_cols]
        transformed = self.preprocessor.transform(row_df)
        best_model = self.models[self.best_model_name]
        pred = best_model.predict(transformed)[0]
        return pred
