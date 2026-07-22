# egnn-mol

E(n)-equivariant graph neural network backbones for molecular simulation, in two flavours that
share the same physics:

- **`E3GNN`** — a dense, native-torch backbone operating on batched padded tensors `(B, N, ·)`.
  Depends only on `torch` + `einops`.
- **`GeometricEGNN`** — a sparse `torch_geometric` backbone operating on packed node tensors
  `(ΣN, ·)` with a `batch` vector, so it handles variable-size (ragged) graph batches.

Both share the same minimum-image periodicity handling, distance encodings, equivariant
coordinate update, and near-identity initialization — the two backbones compute the *same*
function from the same weights (verified by test). The sparse backbone is an **optional**
dependency.

## Install

```bash
pip install egnn-mol            # dense backbone + encodings (torch only)
pip install egnn-mol[pyg]       # + the sparse torch-geometric backbone
```

## Usage

Both backbones expect node features already projected to `dim` (embed atom types / time / etc.
upstream). The equivariant output is the *coordinates*; use their displacement as a velocity.

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

The sparse backbone works on packed tensors and is imported lazily (needs `torch-geometric`):

```python
from egnn_mol import GeometricEGNN   # actionable error if [pyg] is not installed

net = GeometricEGNN(depth=4, dim=64)
x = torch.cat([coors, feats], dim=-1)   # (ΣN, 3 + dim): coordinates then features
out = net(x, edge_index, edge_attr=edge_attr, batch=batch, unitcell_lengths=box_per_node)
coors_out, feats_out = out[:, :3], out[:, 3:]
```

## Hyperparameters

All constructor arguments are keyword-only. `depth` and `dim` are required; everything else has a
default. Arguments not listed on a backbone are forwarded to its layers.

### Common to both backbones

| Name | Default | Description |
|---|---|---|
| `depth` | — | Number of message-passing layers. |
| `dim` | — | Node feature width. Features must already be this wide (embed upstream). |
| `m_dim` | `16` | Hidden message width inside each layer. |
| `edge_dim` | `0` | Per-edge feature width (`0` = no edge features). |
| `dropout` | `0.0` | Dropout probability inside the layer MLPs. |
| `soft_edges` | `False` | Gate each message by a learned scalar in `[0, 1]`. |
| `norm_feats` | `False` | `LayerNorm` node features before the node-feature update. |
| `norm_coors` | `False` | Direction-normalize displacement vectors in the coordinate update (makes the update magnitude box-/bond-length independent). |
| `norm_coors_scale_init` | `1e-2` (dense), `1.0` (sparse) | Initial scale of the coordinate normalizer (only used when `norm_coors=True`). |
| `coor_weights_clamp_value` | `None` | Optional symmetric clamp `[-c, c]` on the per-edge coordinate weights. |
| `distance_cutoff` | `0.0` | If `> 0`, add minimum-image within-cutoff edges to the graph. |
| `tripp_num_layers` | `0` | Depth of the triple-product MLP. `> 0` enables the SE(3) chirality term (see below); `0` keeps the update E(3)-equivariant. |

### Distance encoding

| Name | Default | Description |
|---|---|---|
| `encoding` | `"bessel"` | Radial basis: `"bessel"` (DimeNet orthonormal, implicit cutoff), `"fourier"` (sin/cos bands), `"gaussian"` (fixed-center RBF). |
| `encoding_features` | `8` | Number of basis functions / frequency bands. Output width is this (bessel, gaussian) or `2×` this (fourier). |
| `cutoff` | `10.0` | Radial length scale of the encoding, in the same units as the coordinates. |

### Dense backbone only (`E3GNN`)

Controls how the neighborhood graph is built from coordinates (selected once, shared across layers).

| Name | Default | Description |
|---|---|---|
| `num_nearest_neighbors` | `0` | kNN degree. `0` with no adjacency → dense all-pairs message passing. |
| `only_sparse_neighbors` | `False` | Use exactly the `adj_mat` edges (k = max degree); ignores kNN. |
| `valid_radius` | `inf` | Squared-distance radius below which a neighbor slot counts as a real edge. |
| `m_pool_method` | `"sum"` | How to pool messages onto nodes: `"sum"` or `"mean"`. |

### Sparse backbone only (`GeometricEGNN`)

| Name | Default | Description |
|---|---|---|
| `aggr` | `"sum"` | Message aggregation onto nodes: `"sum"` or `"mean"`. |

### Forward-time inputs (not hyperparameters)

- **`E3GNN.forward`**`(feats (B,N,dim), coors (B,N,3), unitcell_lengths=(B,3)|None, adj_mat=(B,N,N)|(N,N)|None, edges=(B,N,N,edge_dim)|None, mask=(B,N)|None, return_coor_changes=False)` → `(feats, coors)`.
- **`GeometricEGNN.forward`**`(x=(ΣN,3+dim), edge_index=(2,E), edge_attr=(E,edge_dim)|None, batch=(ΣN,)|None, unitcell_lengths=(ΣN,3)|None)` → `(ΣN, 3+dim)`.

## Distance encodings

Encodings are plain functions selected by name. Adding one is a single function plus a `match`
arm in `egnn_mol/encodings.py` — no registry, base class, or factory. `polynomial_envelope` and
`cosine_envelope` provide smooth cutoff weights for bases without an implicit cutoff.

## E(3) vs SE(3)

By default the backbones are E(3)-equivariant. Passing `tripp_num_layers > 0` adds a
triple-product (chirality) term that makes the update SE(3)-equivariant — sensitive to
reflections — which is useful for chiral molecules. Rotation and translation equivariance are
preserved; reflection equivariance is intentionally broken.

## Periodic boundary conditions

Pass `unitcell_lengths` (orthorhombic box lengths) to wrap displacement vectors with the
minimum-image convention. Omit it for open boundaries. Only the coordinate *displacement*
(`coors_out - coors_in`) is periodic — use it as the velocity, not the raw output coordinates.
