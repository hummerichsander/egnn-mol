# egnn-mol

E(n)-equivariant graph neural network backbones for molecular simulation, in two flavours that
share the same physics and the same pytorch-geometric naming (`x`, `pos`, `edge_index`,
`edge_attr`, `batch`):

- **`EGNN`** — a dense, native-torch backbone operating on batched padded tensors `(B, N, ·)`.
  Its module uses only `torch` + `einops`.
- **`GeometricEGNN`** — a sparse `torch_geometric` backbone operating on packed node tensors
  `(ΣN, ·)` with a `batch` vector, so it handles variable-size (ragged) graph batches.

Both share the same minimum-image periodicity handling, distance encodings, equivariant position
update, and near-identity initialization — the two backbones compute the *same* function from the
same weights (verified by test).

## Install

Not on PyPI yet — install from GitHub:

```bash
pip install "git+https://github.com/hummerichsander/egnn-mol.git"
```

This pulls everything (`torch`, `einops`, `torch-geometric`, `torch-cluster`) — both backbones are
always available, no extras to choose. `torch-cluster` compiles against your installed `torch`, so
if your environment builds wheels in isolation, install `torch` first and add `--no-build-isolation`:

```bash
pip install torch
pip install --no-build-isolation "git+https://github.com/hummerichsander/egnn-mol.git"
```

For development, clone and install editable with the dev extra:

```bash
git clone https://github.com/hummerichsander/egnn-mol.git && cd egnn-mol
pip install torch
pip install --no-build-isolation -e ".[dev]"
```

## Usage

Both backbones take node features `x` and positions `pos` separately, and expect `x` already
projected to `dim` (embed atom types / time / etc. upstream). Both **return `(x, pos)`**; the
equivariant output is `pos` — use its displacement as a velocity.

```python
import torch
from egnn_mol import EGNN

net = EGNN(depth=4, dim=64, encoding="bessel", encoding_features=8, cutoff=1.0)
x = torch.randn(2, 10, 64)                               # node features (B, N, dim)
pos = torch.randn(2, 10, 3)                              # positions (B, N, 3)
box = torch.tensor([[2.0, 2.0, 2.0], [2.5, 2.0, 1.8]])   # periodic box lengths, or omit
x_out, pos_out = net(x, pos, box=box)
velocity = pos_out - pos                                 # displacement is the equivariant output
```

The sparse backbone works on packed tensors:

```python
from egnn_mol import GeometricEGNN

net = GeometricEGNN(depth=4, dim=64, distance_cutoff=1.0)
x = torch.randn(100, 64)          # node features (ΣN, dim)
pos = torch.randn(100, 3)         # positions (ΣN, 3)
x_out, pos_out = net(x, pos, batch=batch)   # edge_index optional; radius graph built internally
```

## Forward API

The `forward` methods are the primary API. Both return `(x, pos)`; only `x` and `pos` are
required. Static edges (bonds) are optional inputs; the distance-based (dynamic) graph is
configured at construction (`distance_cutoff` / `num_nearest_neighbors`).

**`EGNN.forward`** — dense padded tensors:

| Argument | Shape | Description |
|---|---|---|
| `x` | `(B, N, dim)` | Node features. |
| `pos` | `(B, N, 3)` | Node positions. |
| `adj_mat` | `(B, N, N)` or `(N, N)`, or `None` | Static bond adjacency (bool). |
| `edge_attr` | `(B, N, N, edge_dim)`, or `None` | Static edge features. |
| `mask` | `(B, N)`, or `None` | Node validity mask (padding). |
| `box` | `(B, 3)`, or `None` | Periodic box lengths (`None` = open boundaries). |
| `return_pos_changes` | bool | Also return the per-layer position trajectory. |

**`GeometricEGNN.forward`** — packed graph tensors:

| Argument | Shape | Description |
|---|---|---|
| `x` | `(ΣN, dim)` | Node features. |
| `pos` | `(ΣN, 3)` | Node positions. |
| `edge_index` | `(2, E)`, or `None` | Static bonds, `[source/neighbor, target/center]`. |
| `edge_attr` | `(E, edge_dim)`, or `None` | Static edge features. |
| `batch` | `(ΣN,)`, or `None` | Graph membership for a ragged batch. |
| `box` | `(ΣN, 3)`, or `None` | Per-node periodic box lengths (`None` = open). |

## Hyperparameters

All constructor arguments are keyword-only and **identical across both backbones**. `depth` and
`dim` are required; everything else has a default. Extra arguments are forwarded to the layers.

### Model

| Name | Default | Description |
|---|---|---|
| `depth` | — | Number of message-passing layers. |
| `dim` | — | Node feature width. Features must already be this wide (embed upstream). |
| `m_dim` | `16` | Hidden message width inside each layer. |
| `edge_dim` | `0` | Static edge-feature width (`0` = no edge features). |
| `aggr` | `"sum"` | Message aggregation onto nodes: `"sum"` or `"mean"`. |
| `dropout` | `0.0` | Dropout probability inside the layer MLPs. |
| `soft_edges` | `False` | Gate each message by a learned scalar in `[0, 1]`. |
| `norm_x` | `False` | `LayerNorm` node features before the node-feature update. |
| `norm_pos` | `False` | Direction-normalize displacement vectors in the position update (makes the update magnitude box-/bond-length independent). |
| `norm_pos_scale_init` | `1.0` | Initial scale of the position normalizer (only used when `norm_pos=True`). |
| `pos_weights_clamp_value` | `None` | Optional symmetric clamp `[-c, c]` on the per-edge position weights. |
| `tripp_num_layers` | `0` | Depth of the triple-product MLP. `> 0` enables the SE(3) chirality term (see below); `0` keeps the update E(3)-equivariant. |

### Neighborhood — static edges ∪ dynamic edges

The graph is the **union** of externally-supplied static bonds and internally-built distance
graphs; self-loops are always excluded. With none of the three below active it is **all-pairs**.

| Name | Default | Description |
|---|---|---|
| static edges | *(forward input)* | Molecular bonds, passed at call time — dense `adj_mat`/`edge_attr`, sparse `edge_index`/`edge_attr`. |
| `distance_cutoff` | `0.0` | If `> 0`, add a **radius graph** (all pairs within the cutoff), built internally from positions. |
| `num_nearest_neighbors` | `0` | If `> 0`, add a **kNN graph**, built internally from positions. |

### Edge features

`edge_dim > 0` gives every edge a length-`edge_dim` feature vector consumed by the edge MLP. You
supply features for **static** edges only:

- **Dense** — `edge_attr` is a dense `(B, N, N, edge_dim)` tensor; the backbone reads
  `edge_attr[b, i, j]` for each edge (center `i`, neighbor `j`) in the neighborhood. Put your bond
  features at the bonded pairs and leave the rest at zero.
- **Sparse** — `edge_attr` is `(E, edge_dim)`, one row per supplied `edge_index` edge.

**Dynamic edges** (radius / kNN) are generated internally from positions, so you do not supply
their features — they get an **all-zero** `edge_dim` vector:

- Sparse: each generated edge is appended with a zero row. If a generated edge coincides with a
  static bond, the duplicate is coalesced (summed), so the bond's features are kept and the zero
  contributes nothing.
- Dense: a generated edge `(i, j)` reads `edge_attr[b, i, j]`; with non-bond entries left at zero
  (the recommended convention, and what makes the two backbones agree) it is zero. You *can*
  populate arbitrary `(i, j)` entries to give specific distance edges their own features, but the
  sparse backbone has no such per-pair channel, so doing this breaks dense/sparse parity.

To let the network distinguish bonds from distance edges, reserve one channel of `edge_dim` as a
**bond indicator**: set it to `1` on your static bonds; dynamic edges read `0` there automatically.

### Distance encoding

| Name | Default | Description |
|---|---|---|
| `encoding` | `"bessel"` | Radial basis: `"bessel"` (DimeNet orthonormal, implicit cutoff), `"fourier"` (sin/cos bands), `"gaussian"` (fixed-center RBF). |
| `encoding_features` | `8` | Number of basis functions / frequency bands. Output width is this (bessel, gaussian) or `2×` this (fourier). |
| `cutoff` | `10.0` | Radial length scale of the encoding, in the same units as the positions. |

Encodings are plain functions selected by name — adding one is a single function plus a `match`
arm in `egnn_mol/encodings.py` (no registry, base class, or factory). `polynomial_envelope` and
`cosine_envelope` provide smooth cutoff weights for bases without an implicit cutoff.

## E(3) vs SE(3)

By default the backbones are E(3)-equivariant. Passing `tripp_num_layers > 0` adds a
triple-product (chirality) term that makes the update SE(3)-equivariant — sensitive to
reflections — which is useful for chiral molecules. Rotation and translation equivariance are
preserved; reflection equivariance is intentionally broken.

## Periodic boundary conditions

Pass `box` (orthorhombic box lengths) to wrap displacement vectors with the minimum-image
convention; omit it for open boundaries. Only the *displacement* (`pos_out - pos`) is periodic —
use it as the velocity, not the raw output positions.

**Dynamic-graph construction** uses `torch_cluster` (`radius_graph`/`knn_graph`) for open
boundaries and a dense minimum-image builder under periodic boundaries (`box` given), since
`torch_cluster` has no PBC support. The dense periodic builder is `O(N²)`; see the roadmap.

## Performance

Dense vs sparse forward latency and **measured** peak memory on random open-boundary systems (fixed
density, radius graph, `depth=4, dim=32`), from `benchmarks/backbones.py` on CPU (indicative —
reproduce with the script or the `benchmark` CI workflow). Memory is measured for real (not
estimated) on both CPU and CUDA by replaying the profiler's allocation events; the benchmark also
attributes peak memory to each message-passing layer.

| N atoms | dense latency | sparse latency | dense peak mem | sparse peak mem | per-layer peak |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.1 ms | 2.1 ms | 0.5 MB | 0.04 MB | 0.02 MB |
| 512 | 15.2 ms | 13.4 ms | 8.1 MB | 0.17 MB | 0.07 MB |
| 1024 | 33.6 ms | 8.7 ms | 32.5 MB | 0.35 MB | 0.14 MB |
| 2048 | 96.8 ms | 18.8 ms | 130.0 MB | 0.70 MB | 0.29 MB |

The sparse backbone scales far better in memory: its cost grows with the number of edges
`E ~ O(N)` at fixed density, while the dense backbone materializes an `O(N²)` distance matrix to
build the neighborhood — its peak memory grows quadratically (0.5 MB → 130 MB from N=128 to 2048).

Notably the **per-layer** peak is identical for both backbones and scales only `O(N)`: inside a
layer both materialize the same `(E, m_dim)` message tensor. The dense backbone's memory
disadvantage is entirely in the `O(N²)` neighborhood construction *outside* the layers, not in the
message passing itself. Use the dense backbone for small, fixed-size systems (simpler batched
tensors); use the sparse backbone for larger systems and ragged (variable-size) batches.

The sparse layer uses a plain scatter message-pass rather than PyG's `MessagePassing`. For
nonlinear EGNN messages the two are performance-equivalent (both materialize `(E, m_dim)` and
scatter); `benchmarks/message_passing.py` confirms **identical outputs** and parity in time
(custom ≈ 0.95× the `MessagePassing` wall-clock, avoiding `propagate()` overhead).

## Roadmap / TODO

- **Fully-periodic scalable neighbor list.** The periodic dynamic-graph path is currently a dense
  `O(N²)` minimum-image pass. A ghost-image expansion (replicate atoms into the 26 adjacent
  periodic images, run `torch_cluster` against them, map image-edges back to base atoms with the
  MIC displacement) would give an `O(N)` periodic radius graph for large boxes (bilayers, solvated
  systems).
