"""Dense EGNN vs sparse GeometricEGNN inference latency and memory across system sizes.

Both backbones run the same radius-graph neighborhood on random open-boundary systems of growing
size (fixed density, so the neighbor count per atom stays roughly constant). Reports median forward
latency, the measured peak memory of a full forward, and the measured peak memory allocated *inside
each message-passing layer*.

Memory is measured (not estimated) on both CPU and CUDA via the PyTorch profiler: allocation events
are replayed in time order to recover the running allocated total, whose maximum is the peak. Each
layer is wrapped in a ``record_function`` scope so the peak can be attributed per layer.
Run: ``python benchmarks/backbones.py``."""

import json
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from egnn_mol import EGNN, GeometricEGNN

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


def _hook_layers(layers: nn.ModuleList) -> list:
    """Wrap each layer's forward in a ``record_function`` scope named ``layer{i}``.

    :param layers: The backbone's message-passing layers.
    :return: The registered hook handles (call ``.remove()`` to detach)."""
    handles, live = [], {}

    def make_pre(i):
        def pre(_mod, _inp):
            rf = record_function(f"layer{i}")
            rf.__enter__()
            live[i] = rf

        return pre

    def make_post(i):
        def post(_mod, _inp, _out):
            live.pop(i).__exit__(None, None, None)

        return post

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre(i)))
        handles.append(layer.register_forward_hook(make_post(i)))
    return handles


def _mem_field(device: torch.device) -> str:
    """Profiler FunctionEvent attribute holding per-op net memory for this device."""
    return (
        "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage"
    )


def _peak_memory(fn, layers: nn.ModuleList, device: torch.device) -> dict:
    """Measured peak memory (MB) of one forward, globally and per layer.

    Replays allocation events in time order: the running sum of per-op net allocations tracks the
    live allocated bytes, and its maximum is the peak. Per-layer peaks are computed over the
    events falling inside each layer's ``record_function`` window.

    :param fn: Zero-arg callable running one forward.
    :param layers: The backbone's message-passing layers (wrapped for attribution).
    :param device: Device the forward runs on.
    :return: ``{"global": MB, "per_layer": [MB, ...]}``."""
    handles = _hook_layers(layers)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, profile_memory=True) as prof:
            fn()
    finally:
        for h in handles:
            h.remove()

    field = _mem_field(device)
    events = [
        (e.time_range.start, e.time_range.end, getattr(e, field), e.name)
        for e in prof.events()
    ]
    allocs = sorted([(s, m) for s, _, m, _ in events if m], key=lambda e: e[0])

    def peak_in(lo: float, hi: float) -> float:
        run = mx = 0
        for start, delta in allocs:
            if lo <= start <= hi:
                run += delta
                mx = max(mx, run)
        return mx / 1e6

    global_peak = peak_in(float("-inf"), float("inf"))
    scopes = {name: (s, en) for s, en, _, name in events if name.startswith("layer")}
    per_layer = [round(peak_in(*scopes[f"layer{i}"]), 3) for i in range(len(layers))]
    return {"global": round(global_peak, 3), "per_layer": per_layer}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = (
        EGNN(depth=DEPTH, dim=DIM, m_dim=M_DIM, distance_cutoff=CUTOFF)
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

        for _ in range(2):  # warmup lazy allocations before profiling
            run_dense()
            run_sparse()

        t_dense = _time_forward(run_dense, device)
        t_sparse = _time_forward(run_sparse, device)
        mem_dense = _peak_memory(run_dense, dense.layers, device)
        mem_sparse = _peak_memory(run_sparse, sparse.layers, device)
        row = {
            "N": n,
            "dense_ms": round(t_dense, 3),
            "sparse_ms": round(t_sparse, 3),
            "dense_peak_mb": mem_dense["global"],
            "sparse_peak_mb": mem_sparse["global"],
            "dense_layer_mb": mem_dense["per_layer"],
            "sparse_layer_mb": mem_sparse["per_layer"],
        }
        rows.append(row)
        print(
            f"N={n:5d}  dense={t_dense:8.2f} ms / {mem_dense['global']:8.2f} MB  "
            f"sparse={t_sparse:8.2f} ms / {mem_sparse['global']:7.2f} MB  "
            f"(dense per-layer peak {max(mem_dense['per_layer']):.2f} MB, "
            f"sparse per-layer peak {max(mem_sparse['per_layer']):.2f} MB)"
        )

    print("\nJSON:")
    print(json.dumps({"device": device.type, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
