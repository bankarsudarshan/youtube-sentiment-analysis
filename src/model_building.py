# src/model_building.py
import logging
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")
fh = logging.FileHandler(os.path.join(log_dir, "model_building.log"))
ch = logging.StreamHandler()
for h in (fh, ch):
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    logger.debug(f"Training data loaded with shape {df.shape}")
    return df


def build_tfidf(train_df: pd.DataFrame, max_features: int, ngram_range: tuple):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_df["clean_comment"])
    y_train = train_df["category"].values
    logger.debug(f"TF-IDF fit. Shape: {X_train.shape}")
    return X_train, y_train, vectorizer


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        is_unbalance=True,
        class_weight="balanced",
        reg_alpha=0.1,
        reg_lambda=0.1,
        **params,
    )
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    logger.debug("LightGBM training completed")
    return model, duration


def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug("Saved object to %s", path)
