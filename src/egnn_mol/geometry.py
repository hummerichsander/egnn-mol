"""Geometric primitives shared by the dense and sparse backbones."""

import torch
from torch import Tensor


def minimum_image(rel_coors: Tensor, box: Tensor | None) -> Tensor:
    """Apply the orthorhombic minimum-image convention to displacement vectors.

    The wrap acts on the trailing spatial axis only and broadcasts over all leading
    axes, so the dense backbone can pass a box of shape (B, 1, 3) and the sparse
    backbone a per-edge box of shape (E, 3) — both hit the same implementation.

    :param rel_coors: Displacement vectors with a trailing spatial axis (..., 3).
    :param box: Box lengths broadcastable to ``rel_coors`` (e.g. (3,), (E, 3), (B, 1, 3));
        None disables periodicity.
    :return: Displacements wrapped into the primary cell, same shape as ``rel_coors``."""

    if box is None:
        return rel_coors
    return rel_coors - box * torch.round(rel_coors / box)


def squared_distance(rel_coors: Tensor) -> Tensor:
    """Squared L2 norm of displacement vectors along the trailing axis.

    :param rel_coors: Displacement vectors (..., 3).
    :return: Squared distances (..., 1)."""

    return (rel_coors**2).sum(dim=-1, keepdim=True)


def signed_volume(v1: Tensor, v2: Tensor, v3: Tensor) -> Tensor:
    """Signed volume det[v1 v2 v3] = (v1 x v2) . v3 of three vector fields.

    This is a pseudoscalar that flips sign under reflection, so feeding it into an
    otherwise E(3)-equivariant update reduces the symmetry to SE(3) (chirality-aware).

    :param v1: First vectors (..., 3).
    :param v2: Second vectors (..., 3).
    :param v3: Third vectors (..., 3).
    :return: Signed volumes (..., 1)."""

    return (torch.cross(v1, v2, dim=-1) * v3).sum(dim=-1, keepdim=True)
