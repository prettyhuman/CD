from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def _macro_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    # one-vs-rest
    y_onehot = np.zeros((y_true.shape[0], n_classes), dtype=int)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    try:
        return float(roc_auc_score(y_onehot, proba, average='macro', multi_class='ovr'))
    except Exception:
        return float('nan')


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    patience: int = 12


def train_model(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                n_classes: int, device: str, cfg: TrainConfig) -> Dict[str, object]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # class weights for imbalance
    counts = np.bincount(y_tr, minlength=n_classes)
    w = (counts.sum() / np.maximum(counts, 1))
    w = w / w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best = float('inf')
    best_state = None
    bad = 0

    def batches(X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for s in range(0, len(X), cfg.batch_size):
            b = idx[s:s+cfg.batch_size]
            yield X[b], y[b]

    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in batches(X_tr, y_tr):
            xb_t = torch.from_numpy(xb).float().to(device)
            yb_t = torch.from_numpy(yb).long().to(device)
            logits = model(xb_t)
            base_loss = loss_fn(logits, yb_t)
            reg_loss = base_loss.new_tensor(0.0)
            if hasattr(model, 'reg_loss') and callable(getattr(model, 'reg_loss')):
                reg_loss = model.reg_loss()
            loss = base_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_va).float().to(device)
            yv = torch.from_numpy(y_va).long().to(device)
            lv = model(xv)
            vloss = loss_fn(lv, yv).item()

        if vloss + 1e-6 < best:
            best = vloss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {'best_val_loss': best, 'epochs_ran': ep+1}


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, n_classes: int, device: str) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        logits = model(xt)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred = proba.argmax(axis=1)

    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average='macro'))
    mauc = _macro_auc(y, proba, n_classes)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    return {'Accuracy': acc, 'MacroF1': mf1, 'MacroAUC': float(mauc), 'confusion_matrix': cm, 'proba': proba, 'pred': pred}
