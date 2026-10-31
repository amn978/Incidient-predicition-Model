"""Model zoo. Every model is a full sklearn Pipeline: raw dataframe -> probabilities.

Keeping feature building INSIDE the pipeline guarantees training and serving use the
exact same code path (no train/serve skew).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .features import CAT_COLS, NUM_COLS, FeatureBuilder


def _preprocessor(kind: str) -> ColumnTransformer:
    """kind='sparse': rich TF-IDF (words + chars) for linear models.
       kind='compact': small TF-IDF + numeric columns for tree models."""
    if kind == "sparse":
        text = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000, sublinear_tf=True)),
        ])
        parts = [("text", text, "text"), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)]
    else:
        text = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=1500, sublinear_tf=True)
        parts = [("text", text, "text"), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                 ("num", "passthrough", NUM_COLS)]
    return ColumnTransformer(parts)


def _pipe(kind: str, clf) -> Pipeline:
    return Pipeline([("features", FeatureBuilder()), ("prep", _preprocessor(kind)), ("clf", clf)])


# ----------------------------- builders -----------------------------------
def build_logreg(seed: int) -> Pipeline:
    return _pipe("sparse", LogisticRegression(C=4.0, max_iter=500, class_weight="balanced", random_state=seed))


def build_linear_svm(seed: int) -> Pipeline:
    svm = LinearSVC(C=0.5, class_weight="balanced", random_state=seed)
    return _pipe("sparse", CalibratedClassifierCV(svm, method="sigmoid", cv=3))


def build_complement_nb(seed: int) -> Pipeline:
    return _pipe("sparse", ComplementNB(alpha=0.3))


def build_random_forest(seed: int) -> Pipeline:
    return _pipe("compact", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=seed))


def _lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.6, class_weight="balanced", random_state=seed,
                          n_jobs=-1, verbose=-1)


def build_lightgbm(seed: int) -> Pipeline:
    return _pipe("compact", _lgbm(seed))


def build_xgboost(seed: int) -> Pipeline:
    return _pipe("compact", XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8,
                                          colsample_bytree=0.6, tree_method="hist", objective="multi:softprob",
                                          n_jobs=-1, random_state=seed, verbosity=0))


def build_ensemble(seed: int) -> VotingClassifier:
    """Soft-vote of a linear text model and a boosted-tree model (different error profiles)."""
    return VotingClassifier([("lr", build_logreg(seed)), ("lgbm", build_lightgbm(seed))], voting="soft")


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Optional semantic embeddings (needs `pip install sentence-transformers`)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer  # lazy import
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(list(X), batch_size=64, show_progress_bar=False))


def build_embeddings_lgbm(seed: int, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Pipeline:
    prep = ColumnTransformer([("emb", SentenceEmbedder(model_name), "text"),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                              ("num", "passthrough", NUM_COLS)])
    return Pipeline([("features", FeatureBuilder()), ("prep", prep), ("clf", _lgbm(seed))])


# ----------------------------- registry -----------------------------------
@dataclass
class ModelSpec:
    name: str
    label: str
    build: Callable[[int], object]
    family: str
    needs_sample_weight: bool = False
    optional: bool = False
    notes: str = field(default="")


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("logreg", "Logistic Regression (TF-IDF)", build_logreg, "linear", notes="Fast, interpretable baseline"),
    ModelSpec("linear_svm", "Linear SVM (TF-IDF, calibrated)", build_linear_svm, "linear", notes="Strong on sparse text"),
    ModelSpec("complement_nb", "Complement Naive Bayes", build_complement_nb, "probabilistic", notes="Imbalance-friendly text baseline"),
    ModelSpec("random_forest", "Random Forest", build_random_forest, "tree", notes="Robust on mixed features"),
    ModelSpec("lightgbm", "LightGBM", build_lightgbm, "boosting", notes="Fast gradient boosting"),
    ModelSpec("xgboost", "XGBoost", build_xgboost, "boosting", needs_sample_weight=True, notes="Gradient boosting"),
    ModelSpec("ensemble_lr_lgbm", "Ensemble (LogReg + LightGBM)", build_ensemble, "ensemble", notes="Soft voting"),
    ModelSpec("embeddings_lgbm", "Sentence embeddings + LightGBM", build_embeddings_lgbm, "semantic", optional=True,
              notes="Semantic text understanding; needs sentence-transformers"),
]}


def available_models(include_optional: bool = False) -> list[str]:
    return [n for n, s in REGISTRY.items() if include_optional or not s.optional]
