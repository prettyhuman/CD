from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLS, LABEL_COL

@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding='utf-8-sig')


def get_feature_matrix(df: pd.DataFrame, drop_constant: bool = True) -> Tuple[np.ndarray, list[str]]:
    X = df[FEATURE_COLS].copy()
    if drop_constant:
        nun = X.nunique(dropna=False)
        keep = nun[nun > 1].index.tolist()
        X = X[keep]
        return X.to_numpy(dtype=float), keep
    return X.to_numpy(dtype=float), FEATURE_COLS


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # first split out test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # split train/val from remaining
    val_ratio_in_tr = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_ratio_in_tr, random_state=seed, stratify=y_tr
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def standardize_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def prepare_splits(
    df: pd.DataFrame,
    use_normal_only_for_graph: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Return splits for diagnosis + a (possibly normal-only) matrix for causal discovery."""
    X_all, feat_names = get_feature_matrix(df, drop_constant=True)
    y_all = df[LABEL_COL].to_numpy(dtype=int)

    X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_val_test(X_all, y_all, seed=seed)
    X_tr, X_va, X_te, scaler = standardize_splits(X_tr, X_va, X_te)

    # causal discovery data
    if use_normal_only_for_graph:
        X_graph = X_all[df[LABEL_COL].to_numpy(dtype=int) == 0]
        X_graph = scaler.transform(X_graph)
    else:
        X_graph = scaler.transform(X_all)

    return {
        'splits': DatasetSplits(X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names, scaler),
        'X_graph': X_graph,
        'feature_names': feat_names,
    }
