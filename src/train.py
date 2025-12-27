from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import os
import joblib

from src.data_loader import load_transactions, generate_sample_transactions
from src.preprocess import clean_data, engineer_features, prepare_dataset
from src.preprocess import clean_data, engineer_features, prepare_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_preprocessor(X):
    # separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")

def train_and_select(X, y, random_state=42):
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    # candidate models
    candidates = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    best_model = None
    best_score = -1
    for name, clf in candidates.items():
        pipe = ImbPipeline([("pre", preprocessor), ("over", RandomOverSampler(random_state=random_state)), ("clf", clf)])
        # quick CV on training set (roc_auc where applicable)
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            mean_score = np.mean(scores)
        except Exception:
            # fallback to accuracy if ROC fails (extreme imbalance)
            scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
            mean_score = np.mean(scores)

        print(f"[train] Model: {name}  CV score(mean)={mean_score:.4f}")

        pipe.fit(X_train, y_train)
        # evaluate on holdout
        print(f"[train] Evaluation for {name} on holdout set:")
        evaluate_model(pipe, X_test, y_test)

        # decide best model by ROC-AUC if available, else by F1 on holdout
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            metric = roc_auc_score(y_test, y_proba)
        except Exception:
            p, r, f1, _ = precision_recall_fscore_support(y_test, pipe.predict(X_test), y_test, average="binary", zero_division=0)
            metric = f1

        if metric > best_score:
            best_score = metric
            best_model = (name, pipe)

    # Save best
    if best_model is not None:
        name, model_pipe = best_model
        model_path = os.path.join(MODEL_DIR, "best_model.joblib")
        joblib.dump(model_pipe, model_path)
        print(f"[train] Saved best model ({name}) to: {model_path}")
    else:
        raise RuntimeError("No model selected")

    return best_model

if __name__ == '__main__':
    # If data missing, generate sample
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
    if not os.path.exists(data_path):
        print("[train] No data found; generating synthetic dataset (n=10000)")
        generate_sample_transactions(path=data_path, n=10000)

    df = load_transactions(path=data_path)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = prepare_dataset(df)

    # we may have lots of zero-variance cols after one-hot; drop constant cols
    X = X.loc[:, X.apply(pd.Series.nunique) > 1]

    best = train_and_select(X, y)
    print('[train] Done.')
