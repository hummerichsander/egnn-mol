"""Dense E3GNN vs sparse GeometricEGNN inference latency and memory across system sizes.

Both backbones run the same radius-graph neighborhood on random open-boundary systems of growing
size (fixed density, so the neighbor count per atom stays roughly constant). Reports median forward
latency and peak memory (measured on CUDA; on CPU the dominant intermediate allocation is reported
as an estimate, since the dense backbone builds an (N, N) distance matrix while the sparse backbone
stays at O(E)). Run: ``python benchmarks/backbones.py``."""

import json
import time

import torch

from egnn_mol import E3GNN, GeometricEGNN

SIZES = [64, 128, 256, 512, 1024, 2048]
DIM, DEPTH, M_DIM, CUTOFF, DENSITY = (
    32,
    4,
    32,
    1.5,
    0.3,
)  # ~ constant neighbors per atom


def _random_system(n: int, device: torch.device):
    """A cubic open-boundary system of ``n`` atoms at fixed number density."""
    side = (n / DENSITY) ** (1 / 3)
    pos = torch.rand(n, 3, device=device) * side
    x = torch.randn(n, DIM, device=device)
    return x, pos


def _time_forward(fn, device: torch.device, iters: int = 20) -> float:
    """Median forward latency (ms), with a few warmup runs."""
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _peak_mb(fn, device: torch.device) -> float | None:
    """Measured peak CUDA memory (MB) for one forward, or None on CPU."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = (
        E3GNN(depth=DEPTH, dim=DIM, m_dim=M_DIM, distance_cutoff=CUTOFF)
        .eval()
        .to(device)
    )
    sparse = (
        GeometricEGNN(depth=DEPTH, dim=DIM, m_dim=M_DIM, distance_cutoff=CUTOFF)
        .eval()
        .to(device)
    )

    rows = []
    torch.manual_seed(0)
    for n in SIZES:
        x, pos = _random_system(n, device)

        def run_dense():
            with torch.no_grad():
                return dense(x[None], pos[None])

        def run_sparse():
            with torch.no_grad():
                return sparse(x, pos)

        t_dense = _time_forward(run_dense, device)
        t_sparse = _time_forward(run_sparse, device)
        # Dominant intermediate: dense builds an (N, N) float distance matrix; sparse stays O(E).
        dense_distmat_mb = n * n * 4 / 1e6
        row = {
            "N": n,
            "dense_ms": round(t_dense, 3),
            "sparse_ms": round(t_sparse, 3),
            "dense_peak_mb": _peak_mb(run_dense, device),
            "sparse_peak_mb": _peak_mb(run_sparse, device),
            "dense_distmat_mb_est": round(dense_distmat_mb, 2),
        }
        rows.append(row)
        print(
            f"N={n:5d}  dense={t_dense:8.2f} ms  sparse={t_sparse:8.2f} ms  "
            f"(dense N^2 distance matrix ~ {dense_distmat_mb:.1f} MB)"
        )

    print("\nJSON:")
    print(json.dumps({"device": device.type, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
