"""GNN models for fault diagnosis.

One model per file for clarity:
- gin.py        : GIN
- graphsage.py  : GraphSAGE
- gat.py        : GAT
- graphormer.py : Graphormer (lightweight)
- cignn.py      : Ours (CI-GNN-style)

Scripts typically call `build_model(...)`.
"""

from .factory import build_model
from .gin import GINClassifier
from .graphsage import GraphSAGEClassifier
from .gat import GATClassifier
from .graphormer import GraphormerClassifier
from .cignn import CIGNNClassifier

__all__ = [
    'build_model',
    'GINClassifier',
    'GraphSAGEClassifier',
    'GATClassifier',
    'GraphormerClassifier',
    'CIGNNClassifier',
]
