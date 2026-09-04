# egnn-mol

E(n)-equivariant graph neural network backbones for molecular simulation, in two flavours that
share the same physics and naming — `x` = node **positions**, `h_node` = node features,
`h_edge` = edge features, `edge_index` = connectivity, `batch` = graph membership:

- **`EGNN`** — a dense, native-torch backbone operating on batched padded tensors `(B, N, ·)`.
  Its module uses only `torch` + `einops`.
- **`GeometricEGNN`** — a sparse `torch_geometric` backbone operating on packed node tensors
  `(ΣN, ·)` with a `batch` vector, so it handles variable-size (ragged) graph batches.

Both share the same minimum-image periodicity handling, distance encodings, equivariant position
update, and near-identity initialization — the two backbones compute the *same* function from the
same weights (verified by test).

Alongside them:

- **`RadialField`** — a sparse velocity field with an **analytic divergence**, after Köhler, Klein
  & Noé ([arXiv:2006.02425](https://arxiv.org/abs/2006.02425)). It returns `(v, div v)` instead of
  `(h_node, x)`, trading the EGNN's message passing for a closed-form Jacobian trace — which is
  what a continuous normalizing flow needs to compute an exact log-likelihood without `O(3N)`
  backward passes or a stochastic trace estimator. It shares the encodings, the periodicity
  handling and the neighborhood builder with `GeometricEGNN`.

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

Both backbones take node features `h_node` and positions `x` separately, and expect `h_node`
already projected to `dim` (embed atom types / time / etc. upstream). Both **return `(h_node, x)`**;
the equivariant output is `x` — use its displacement as a velocity.

```python
import torch
from egnn_mol import EGNN

net = EGNN(depth=4, dim=64, encoding="bessel", encoding_features=8, cutoff=1.0)
h_node = torch.randn(2, 10, 64)  # node features (B, N, dim)
x = torch.randn(2, 10, 3)  # positions (B, N, 3)
box = torch.tensor([[2.0, 2.0, 2.0], [2.5, 2.0, 1.8]])  # periodic box lengths, or omit
h_node_out, x_out = net(h_node, x, box=box)
velocity = x_out - x  # displacement is the equivariant output
```

The sparse backbone works on packed tensors:

```python
from egnn_mol import GeometricEGNN

net = GeometricEGNN(depth=4, dim=64, distance_cutoff=1.0)
h_node = torch.randn(100, 64)  # node features (ΣN, dim)
x = torch.randn(100, 3)  # positions (ΣN, 3)
h_node_out, x_out = net(
    h_node, x, batch=batch
)  # edge_index optional; radius graph built internally
```

`RadialField` is a velocity field rather than a layer stack, so it returns the velocity together
with its exact divergence — one scalar per graph. Its `forward` signature is otherwise identical to
`GeometricEGNN`'s, time included: a flow writes `t` into an extra node-feature channel and widens
`dim` by one.

```python
from egnn_mol import RadialField

net = RadialField(
    dim=65, encoding="bessel", encoding_features=32, cutoff=1.0, distance_cutoff=1.0
)
h_node = torch.randn(100, 64)  # node features (ΣN, dim - 1)
x = torch.randn(100, 3)  # positions (ΣN, 3)
t = torch.full((100, 1), 0.5)  # time, as one more node feature

# v: (ΣN, 3), div: (num_graphs,)
v, div = net(torch.cat([h_node, t], -1), x, batch=batch)
```

`div` equals the autograd trace of `dv/dx` to machine precision (`tests/test_radial.py`), so it
drops straight into `d/dt log p = -div v` with no Hutchinson estimator.

## Forward API

The `forward` methods are the primary API. The two EGNNs return `(h_node, x)` and need only
`h_node` and `x`; `RadialField` returns `(v, div v)` and additionally needs `t`. Static edges
(bonds) are optional inputs; the distance-based (dynamic) graph is configured at construction
(`distance_cutoff` / `num_nearest_neighbors`). The graph is rebuilt from positions on every
call, so **`distance_cutoff` may also be passed per call** to resize the neighborhood without
building a new module — `num_nearest_neighbors` stays construction-only.

**`EGNN.forward`** — dense padded tensors:

| Argument | Shape | Description |
|---|---|---|
| `h_node` | `(B, N, dim)` | Node features. |
| `x` | `(B, N, 3)` | Node positions. |
| `adj_mat` | `(B, N, N)` or `(N, N)`, or `None` | Static bond adjacency (bool). |
| `h_edge` | `(B, N, N, edge_dim)`, or `None` | Static edge features. |
| `mask` | `(B, N)`, or `None` | Node validity mask (padding). |
| `box` | `(B, 3)`, or `None` | Periodic box lengths (`None` = open boundaries). |
| `return_x_changes` | bool | Also return the per-layer position trajectory. |
| `distance_cutoff` | float, or `None` | Radius of the dynamic graph for this call, overriding the constructed one. The envelope tapers at the same radius. |

**`GeometricEGNN.forward`** — packed graph tensors:

| Argument | Shape | Description |
|---|---|---|
| `h_node` | `(ΣN, dim)` | Node features. |
| `x` | `(ΣN, 3)` | Node positions. |
| `edge_index` | `(2, E)`, or `None` | Static bonds, `[source/neighbor, target/center]`. |
| `h_edge` | `(E, edge_dim)`, or `None` | Static edge features. |
| `batch` | `(ΣN,)`, or `None` | Graph membership for a ragged batch. |
| `box` | `(ΣN, 3)`, or `None` | Per-node periodic box lengths (`None` = open). |
| `distance_cutoff` | float, or `None` | Radius of the dynamic graph for this call, overriding the constructed one. The envelope tapers at the same radius. |

**`GeometricEGNN.forward_and_divergence`** — the same arguments, returning
`(h_node, x, div)` where `div` is the exact divergence of the displacement field `x_out - x`,
one value per graph. Takes one extra argument:

| Argument | Shape | Description |
|---|---|---|
| `create_graph` | bool | Build the second-order graph so the divergence is itself differentiable. |
| `distance_cutoff` | float, or `None` | As above. The colouring is taken from the edges this radius produced, so it follows on its own. |

`sparsity_pattern` and `jacobian_colouring` take `distance_cutoff` too, so a radius can be
priced before it is run at.

**`RadialField.forward`** — packed graph tensors plus a time, returning `(v, div v)`:

| Argument | Shape | Description |
|---|---|---|
| `h_node` | `(ΣN, dim)` | Node features. |
| `x` | `(ΣN, 3)` | Node positions. |
| `t` | `(ΣN, 1)` or `(ΣN,)` | Per-node time, fed to the coefficient head. |
| `edge_index` | `(2, E)`, or `None` | Static bonds, `[source/neighbor, target/center]`. |
| `h_edge` | `(E, edge_dim)`, or `None` | Static edge features. |
| `batch` | `(ΣN,)`, or `None` | Graph membership for a ragged batch. |
| `box` | `(ΣN, 3)`, or `None` | Per-node periodic box lengths (`None` = open). |
| `distance_cutoff` | float, or `None` | As above; the envelope **and its derivative** taper at that radius, so the closed form keeps matching the field. There is no separate envelope flag here, so `0.0` drops the radius graph and its taper together. |

## Hyperparameters

All constructor arguments are keyword-only and **identical across both EGNN backbones**. `depth`
and `dim` are required; everything else has a default. Extra arguments are forwarded to the layers.
`RadialField` shares the encoding and neighborhood arguments and has its own model section below.

### Model — `EGNN` / `GeometricEGNN`

| Name | Default | Description |
|---|---|---|
| `depth` | — | Number of message-passing layers. |
| `dim` | — | Node feature width, time channel included. Features must already be this wide (embed upstream). |
| `m_dim` | `16` | Hidden message width inside each layer. |
| `mlp_depth` | `1` | Hidden blocks in each layer's edge, node and position MLPs. Buys capacity per layer rather than per hop: unlike `depth` and `tripp_num_layers` it does **not** change the receptive field or the Jacobian sparsity pattern. |
| `edge_dim` | `0` | Static edge-feature width (`0` = no edge features). |
| `aggr` | `"sum"` | Message aggregation onto nodes: `"sum"` or `"mean"`. |
| `dropout` | `0.0` | Dropout probability inside the layer MLPs. |
| `soft_edges` | `False` | Gate each message by a learned scalar in `[0, 1]`. |
| `norm_h_node` | `False` | `LayerNorm` node features before the node-feature update. |
| `norm_displacement` | `False` | Direction-normalize displacement vectors in the position update (makes the update magnitude box-/bond-length independent). |
| `norm_displacement_scale_init` | `1.0` | Initial scale of the `DisplacementNorm` (only used when `norm_displacement=True`). |
| `x_weights_clamp_value` | `None` | Optional symmetric clamp `[-c, c]` on the per-edge position weights. |
| `tripp_num_layers` | `0` | Depth of the triple-product MLP. `> 0` enables the SE(3) chirality term (see below); `0` keeps the update E(3)-equivariant. Note it **doubles the receptive field** (see *Sparsity and the exact divergence*). |
| `envelope` | `False` | Taper every edge's contribution to zero at `distance_cutoff`, keeping the field C¹ where an edge enters or leaves the radius graph. Off by default: a network trained without it computes a different function, so enabling it changes what an existing checkpoint evaluates. |
| `envelope_exponent` | `6` | Exponent of that polynomial envelope. |

### `RadialField`

`v_i = Σ_j φ(d_ij, t) r_ij` with `r_ij = x_i - x_j`, whose divergence is
`Σ_ij [∂φ/∂d_ij · d_ij + D · φ]`. `φ` is **linear in the radial basis** — the coefficient head
predicts the basis weights, never `φ` itself — which is what keeps `∂φ/∂d` an exact closed form
rather than another autograd pass. Since the derivation only touches the diagonal Jacobian blocks
`∂v_i/∂x_i`, `φ` may be conditioned on anything that does not read positions: node and edge
features enter freely, and time rides in as one more node feature.

| Name | Default | Description |
|---|---|---|
| `dim` | — | Node feature width, time channel included. Features must already be this wide (embed upstream). |
| `encoding_features` | `16` | Basis width. With no `depth` to stack, this is the primary capacity knob. |
| `m_dim` | `64` | Hidden width of the coefficient head. |
| `head_depth` | `2` | Hidden blocks in the coefficient head. It never reads positions, so it may be arbitrarily deep without touching the closed form. |
| `envelope_exponent` | `6` | Exponent of the polynomial envelope applied when `distance_cutoff > 0`. |

`encoding`, `cutoff`, `edge_dim`, `distance_cutoff` and `num_nearest_neighbors` mean exactly what
they do for the two EGNNs. Two arguments are **absent by construction**:

- **`depth`** — stacking these updates would make the total Jacobian a *product*, whose
  log-determinant is not a sum of traces. The single pairwise sum is the whole receptive field, so
  `distance_cutoff` matters far more here than for a deep EGNN.
- **`aggr`** — the divergence formula above is written for a sum.
- **`t`** — time is a node feature like any other, which is what keeps this `forward` a drop-in
  replacement for `GeometricEGNN`'s.

With `distance_cutoff > 0` the polynomial envelope is applied automatically at that radius, so `φ`
*and* `∂φ/∂d` vanish where edges enter and leave the graph. Without it the field would be
discontinuous at the cutoff and the divergence would only hold away from the boundary. A static
edge set needs no envelope: it does not depend on positions, so the field is smooth everywhere.

### Neighborhood — static edges ∪ dynamic edges

The graph is the **union** of externally-supplied static bonds and internally-built distance
graphs; self-loops are always excluded. With none of the three below active it is **all-pairs**.

| Name | Default | Description |
|---|---|---|
| static edges | *(forward input)* | Molecular bonds, passed at call time — dense `adj_mat`/`h_edge`, sparse `edge_index`/`h_edge`. |
| `distance_cutoff` | `0.0` | If `> 0`, add a **radius graph** (all pairs within the cutoff), built internally from positions. Overridable per call — see *Forward API*. |
| `num_nearest_neighbors` | `0` | If `> 0`, add a **kNN graph**, built internally from positions. |

### Edge features

`edge_dim > 0` gives every edge a length-`edge_dim` feature vector consumed by the edge MLP. You
supply features for **static** edges only:

- **Dense** — `h_edge` is a dense `(B, N, N, edge_dim)` tensor; the backbone reads
  `h_edge[b, i, j]` for each edge (center `i`, neighbor `j`) in the neighborhood. Put your bond
  features at the bonded pairs and leave the rest at zero.
- **Sparse** — `h_edge` is `(E, edge_dim)`, one row per supplied `edge_index` edge.

**Dynamic edges** (radius / kNN) are generated internally from positions, so you do not supply
their features — they get an **all-zero** `edge_dim` vector:

- Sparse: each generated edge is appended with a zero row. If a generated edge coincides with a
  static bond, the duplicate is coalesced (summed), so the bond's features are kept and the zero
  contributes nothing.
- Dense: a generated edge `(i, j)` reads `h_edge[b, i, j]`; with non-bond entries left at zero
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

Every basis also ships its analytic derivative — `encode_distance_derivative`, plus
`bessel_derivative` / `fourier_derivative` / `gaussian_derivative` and
`polynomial_envelope_derivative` — which is what `RadialField` differentiates instead of calling
autograd. A new encoding therefore needs both arms to be usable there. Note the bessel derivative
carries a Taylor branch below `z = freq·d ≈ 0.1`: `z cos z - sin z` is a difference of two `O(z)`
terms whose result is `O(z³/3)`, so the direct form keeps nothing of it in float32.

## Sparsity and the exact divergence

The edge set is built **once** per forward pass, by neighbor searches that return indices, and is
then shared by every layer. Within one call the graph is therefore fixed, and `dx_out_i / dx_j`
vanishes unless `j` is within `receptive_hops` of `i`:

```
receptive_hops = depth                 # E(3)
receptive_hops = 2 * depth             # SE(3), tripp_num_layers > 0
```

A layer's feature update reads one hop, and so does its position update — unless the chirality
term is on, in which case `x_weight` also sees `chi` at both endpoints, and `chi` is itself a
one-hop aggregate. That is where the factor of two comes from.

That pattern is what makes the trace of the Jacobian cheap. Colour the `receptive_hops`-hop
closure so that no two nodes of a colour are within reach of each other, and one backward pass
per (colour × axis) reads every diagonal block of that colour at once: the off-diagonal terms
sharing a row are structurally zero, so they cannot contaminate the read-out. The count of
colours tracks the size of a `receptive_hops` ball, not the system, so the cost stops scaling
with `N`:

```python
net = GeometricEGNN(depth=2, dim=32, distance_cutoff=0.5, envelope=True, ...)

colours = net.jacobian_colouring(x, edge_index, batch)   # (ΣN,) node groups
h, x_out, div = net.forward_and_divergence(h_node, x, edge_index, batch=batch)
```

`3 * colours` backward passes instead of `3N`, and exact — unlike a Hutchinson estimate, whose
noise survives exponentiation and biases any density built from it. `sparsity_pattern` returns
the closure itself if you want to inspect or measure it.

Two things to get right:

- **Set `envelope=True`.** Without it an edge's contribution is `O(1)` at the cutoff, so the
  field jumps as the edge set changes and there is no well-defined divergence to compute. The
  envelope is evaluated once on the *input* positions and shared by every layer, exactly as the
  edge set is; tapering each layer by its own distances instead reintroduces the jump from
  depth 2 onward, because an edge that entered with zero weight has moved by the time the second
  layer reads it.
- **Never reuse a colouring across an edge-set change.** The graph is rebuilt from positions on
  every call, and a newly-formed edge can put two same-coloured nodes within reach, silently
  corrupting the trace. `forward_and_divergence` recolours per call for exactly this reason —
  which is also why a per-call `distance_cutoff` needs no care here: the colouring is taken from
  the edges that radius produced, so it widens with it. Pass the same radius to
  `sparsity_pattern` to price it first, since a wider ball costs more colours.

### Separating the depth on dynamic and static edges

The ball above is a ball in *space*, `receptive_hops * distance_cutoff` wide, so its colour count
grows cubically with depth — which is what forces a colourable backbone to stay shallow. `dynamic_
layers` breaks that link by naming the layers that see the radius graph; every other layer reads the
caller's static graph alone:

```python
net = GeometricEGNN(depth=9, dim=32, distance_cutoff=0.4, dynamic_layers=(0,), envelope=True, ...)
```

A static hop spreads along the bonds instead of through space, and on a molecular system it cannot
leave the molecule at all, so it is far cheaper: on a 10 000-atom hexadecane box, 32 static hops
colour in 50 groups (200× fewer passes than the dense trace) while a single 0.4 nm dynamic hop
already costs 16 on its own. Nine layers as `(8 static, 1 dynamic)` cost what two all-dynamic
layers do.

The sparsity pattern is then no longer a power of one adjacency but a composition of what each
layer actually reads — `composed_closure` rather than `hop_closure` — which `sparsity_pattern` and
`forward_and_divergence` build for you. Two consequences:

- `receptive_hops` still counts the hops walked, but they are hops of different graphs, so it
  becomes an upper bound on the reach through the full neighborhood rather than the reach itself.
- Where in the stack the dynamic layers sit does not change the cost (the closure is symmetrized,
  and the support of a product of symmetric graphs is order-invariant), only what the network can
  express. `dynamic_layers=()` drops the radius graph from the field entirely.

## E(3) vs SE(3)

By default the backbones are E(3)-equivariant. Passing `tripp_num_layers > 0` adds a
triple-product (chirality) term that makes the update SE(3)-equivariant — sensitive to
reflections — which is useful for chiral molecules. Rotation and translation equivariance are
preserved; reflection equivariance is intentionally broken.

## Periodic boundary conditions

Pass `box` (orthorhombic box lengths) to wrap displacement vectors with the minimum-image
convention; omit it for open boundaries. Only the *displacement* (`x_out - x`) is periodic —
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
- **Dense `RadialField`.** The closed form is layout-agnostic (a per-edge scalar summed per
  graph), but only the sparse form exists, since a dense `build_neighborhood` materializes the
  `(B, N, N)` distance matrix the field has no other use for.
- **Learnable RBF centers and bandwidths.** The original kernel flow optimizes the Gaussian means
  and widths alongside the mixing weights and reports the bandwidths as the knob controlling the
  dynamics' complexity; `gaussian` here has fixed linearly-spaced centers.
