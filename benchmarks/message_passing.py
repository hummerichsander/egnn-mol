"""Confirm the custom scatter message-pass matches a PyG ``MessagePassing`` layer.

For nonlinear EGNN messages both materialize the ``(E, m_dim)`` message tensor and scatter it —
PyG's fused SpMM path does not apply — so they should be output-identical and performance-close,
with the custom path avoiding ``propagate()`` overhead. Run: ``python benchmarks/message_passing.py``."""

import time

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class CustomMP(nn.Module):
    """Manual gather + edge MLP + scatter (the approach used by SparseEGNNLayer)."""

    def __init__(self, dim: int, m_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim, m_dim), nn.SiLU(), nn.Linear(m_dim, m_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = self.edge_mlp(torch.cat([x[dst], x[src]], dim=-1))
        return scatter(m, dst, dim=0, dim_size=x.shape[0], reduce="sum")


class PyGMP(MessagePassing):
    """The same computation routed through ``MessagePassing.propagate``."""

    def __init__(self, dim: int, m_dim: int) -> None:
        super().__init__(aggr="sum", flow="source_to_target")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim, m_dim), nn.SiLU(), nn.Linear(m_dim, m_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j], dim=-1))


def _time(fn, iters: int = 50) -> float:
    """Median wall-clock (ms) of ``iters`` forward+backward passes."""
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        out.sum().backward()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    torch.manual_seed(0)
    n, dim, m_dim, k = 2000, 64, 64, 16
    x = torch.randn(n, dim, requires_grad=True)
    # a directed kNN-like graph: each node points to k others
    dst = torch.arange(n).repeat_interleave(k)
    src = torch.randint(0, n, (n * k,))
    edge_index = torch.stack([src, dst])

    custom, pyg = CustomMP(dim, m_dim), PyGMP(dim, m_dim)
    pyg.edge_mlp.load_state_dict(custom.edge_mlp.state_dict())  # share weights

    with torch.no_grad():
        max_diff = (custom(x, edge_index) - pyg(x, edge_index)).abs().max().item()
    print(f"max output difference: {max_diff:.2e}")
    assert max_diff < 1e-6, "custom and MessagePassing outputs disagree"

    t_custom = _time(lambda: custom(x, edge_index))
    t_pyg = _time(lambda: pyg(x, edge_index))
    print(f"N={n}, E={n * k}, dim={dim}")
    print(f"  custom scatter    : {t_custom:.3f} ms / fwd+bwd")
    print(f"  MessagePassing    : {t_pyg:.3f} ms / fwd+bwd")
    print(f"  custom / PyG ratio: {t_custom / t_pyg:.2f}x")


if __name__ == "__main__":
    main()
