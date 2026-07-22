"""E(n)-equivariant GNN backbones for molecular simulation.

The dense backbone (:class:`E3GNN`) and all encodings work with torch only. The sparse backbone
(:class:`GeometricEGNN`) needs ``torch-geometric`` and is imported lazily, so ``import egnn_mol``
never fails without the ``[pyg]`` extra installed."""

import importlib
import importlib.util
from typing import TYPE_CHECKING

from .dense import E3GNN, E3GNNPeriodic, PeriodicE3GNN
from .encodings import (
    Encoding,
    bessel,
    cosine_envelope,
    encode_distance,
    encoding_width,
    fourier,
    gaussian,
    polynomial_envelope,
)
from .geometry import minimum_image, signed_volume, squared_distance
from .nn import MLP, CoorsNorm
from .update import EquivariantUpdate

if TYPE_CHECKING:
    from .sparse import GeometricEGNN, SparseEGNNLayer, knn_graph_pbc, radius_graph_pbc

__all__ = [
    "E3GNN",
    "E3GNNPeriodic",
    "PeriodicE3GNN",
    "GeometricEGNN",
    "SparseEGNNLayer",
    "radius_graph_pbc",
    "knn_graph_pbc",
    "EquivariantUpdate",
    "MLP",
    "CoorsNorm",
    "minimum_image",
    "squared_distance",
    "signed_volume",
    "Encoding",
    "encode_distance",
    "encoding_width",
    "bessel",
    "fourier",
    "gaussian",
    "polynomial_envelope",
    "cosine_envelope",
    "has_pyg",
]

_LAZY = {
    "GeometricEGNN": "sparse",
    "SparseEGNNLayer": "sparse",
    "radius_graph_pbc": "sparse",
    "knn_graph_pbc": "sparse",
}


def has_pyg() -> bool:
    """Whether ``torch-geometric`` is installed (i.e. whether the sparse backbone is available)."""
    return importlib.util.find_spec("torch_geometric") is not None


def __getattr__(name: str) -> object:
    """Lazily import sparse-backbone symbols so torch-geometric stays optional (PEP 562)."""
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
