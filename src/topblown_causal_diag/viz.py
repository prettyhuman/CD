from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_dag(A: np.ndarray, names: List[str], out_png: str, title: str = '') -> None:
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                G.add_edge(i, j)

    plt.figure(figsize=(12, 10))
    # Circular layout helps readability and is consistent across runs.
    # Keep node order fixed to the feature order.
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=18, width=1.6)
    nx.draw_networkx_labels(G, pos, labels={i:names[i] for i in range(len(names))}, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_png: str,
    title: str = 'Confusion Matrix',
    normalize: str = 'true',
    show_counts: bool = False,
):
    """Plot confusion matrix.

    normalize:
        - 'true': row-normalized (each row sums to 1)
        - 'pred': column-normalized
        - 'all': global-normalized
        - 'none': raw counts
    """
    cm = np.asarray(cm, dtype=np.float32)
    cm_disp = cm.copy()
    if normalize == 'true':
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm_disp = cm / denom
    elif normalize == 'pred':
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        cm_disp = cm / denom
    elif normalize == 'all':
        s = cm.sum()
        cm_disp = cm / (s if s > 0 else 1.0)
    elif normalize == 'none':
        cm_disp = cm
    else:
        raise ValueError(f"Unknown normalize={normalize}")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_disp, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    vmax = float(cm_disp.max()) if cm_disp.size else 0.0
    thresh = vmax * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize == 'none':
                text = f"{int(cm[i, j])}"
            else:
                pct = 100.0 * float(cm_disp[i, j])
                if show_counts:
                    text = f"{pct:.1f}%\n({int(cm[i, j])})"
                else:
                    text = f"{pct:.1f}%"
            plt.text(
                j,
                i,
                text,
                ha='center',
                va='center',
                color='white' if float(cm_disp[i, j]) > thresh else 'black',
                fontsize=8,
            )

    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
