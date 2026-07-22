# egnn-mol

E(n)-equivariant graph neural network backbones for molecular simulation, in two flavours that
share the same physics:

- **`E3GNN`** — a dense, native-torch backbone operating on batched padded tensors `(B, N, ·)`.
  Depends only on `torch` + `einops`.
- **`GeometricEGNN`** — a sparse `torch_geometric` backbone operating on packed node tensors
  `(ΣN, ·)` with a `batch` vector, so it handles variable-size (ragged) graph batches.

Both share the same minimum-image periodicity handling, distance encodings, equivariant
coordinate update, and near-identity initialization. The sparse backbone is an **optional**
dependency.

## Install

```bash
pip install egnn-mol            # dense backbone + encodings (torch only)
pip install egnn-mol[pyg]       # + the sparse torch-geometric backbone
```

## Distance encodings

Radial distance encodings are plain functions selected by name — `"bessel"` (DimeNet orthonormal
Bessel basis), `"fourier"` (sinusoidal bands), `"gaussian"` (fixed-center RBF). Adding one is a
single function plus a `match` arm in `egnn_mol/encodings.py`.

## Usage

```python
import torch
from egnn_mol import E3GNN

net = E3GNN(depth=4, dim=64, encoding="bessel", encoding_features=8, cutoff=1.0)
feats = torch.randn(2, 10, 64)
coors = torch.randn(2, 10, 3)
box = torch.tensor([[2.0, 2.0, 2.0], [2.5, 2.0, 1.8]])   # periodic box lengths, or omit
feats_out, coors_out = net(feats, coors, unitcell_lengths=box)
velocity = coors_out - coors                              # displacement is the equivariant output
```

The sparse backbone is imported lazily; it requires `torch-geometric`:

```python
from egnn_mol import GeometricEGNN   # raises an actionable error if [pyg] is not installed
```
