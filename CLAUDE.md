# Python Style & Conventions

### Imports

Group imports in this order, separated by blank lines:
1. Standard library (`from typing import ...`, `from abc import ...`, `logging`, `os`)
2. Third-party (`torch`, `numpy`, `sklearn`, …)
3. Internal (`from mypackage.utils...`, `from mypackage.models...`)

Always import frequently-used types explicitly rather than accessing them through the module:

```python
from typing import Any, Literal
import logging

import torch
from torch import Tensor, nn
```

### Type Annotations

- Use Python 3.10+ union syntax: `str | None`, `float | None` — never `Optional[str]`
- Use lowercase built-in generics: `dict[str, Tensor]`, `list[int]`, `tuple[Tensor, ...]`
- Annotate every public function signature (parameters and return type)
- Use `Literal[...]` for string-valued enum parameters

### Docstrings

Every public class, method, and function gets a docstring. Format:

```python
def foo(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Short one-line description.

    :param x: Description of x.
    :param t: Description of t.
    :return: Description of the return value."""
```

- One-line summary first, then blank line, then `:param`/`:return:` lines
- Always end sentences with a period
- Class docstrings: one short line is enough

### Naming

- `snake_case` for all variables, functions, and modules
- Single uppercase letters for matrices: `M`, `R`, `Q`
- Short tensor names in local scope are fine: `B`, `N`, `d`, `b`, `n`

### Class Structure

- Define config/Hparams classes directly before the class that uses them
- `__init__` first, then core methods (`forward`, `velocity`, `compute_metrics`), then helpers and properties
- Use short section comments in `__init__` to separate logical blocks
- Use `@property` for computed read-only attributes
- Use `ABC` + `@abstractmethod` for base classes; abstract methods raise `NotImplementedError`

### Modern Python Idioms

- Walrus operator for conditional assignments:
  ```python
  if (value := some_dict.get("key")) is not None:
      ...
  ```
- `match`/`case` for dispatching on string literals
- f-strings for all string formatting

### PyTorch Patterns

- Prefer `torch.cat` / `torch.einsum` over manual loops
- Use `nn.Sequential` for MLPs, `nn.ModuleList` for variable-depth layer stacks
- Use `einops.rearrange` / `einops.repeat` for readable shape manipulation

### Comments

Write comments only when the *why* is non-obvious: a hidden constraint, a numerical workaround, or a subtle invariant. One short line is enough — no multi-line blocks.

```python
# clamp before sampling — numerical solvers can return tiny negative values.
w = torch.clamp(w, min=0.0)
```

Avoid using comments to separate logical blocks of code. Use a  blank line instead.

### Testing

- Use `pytest` with fixtures in `conftest.py`
- Fixture and test-class docstrings follow the same `:param`/`:return:` format
- Test implementations of abstract classes are self-contained in the test file

## Tooling

- **Linter/formatter**: `ruff` (configured in `pyproject.toml`; excludes `tests/`, `.venv/`)
- **Package manager**: `uv`
- **Python**: 3.10–3.12

### Visualization
For visualization use matplotlib with the scienceplots theme:

```python
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature"])
```

The figure size should be taken from sciencplots: 3.3 x 2.5 inches for single-column figures, 6.9 x 2.5 inches for double-column figures:

```python
plt.figure(figsize=(3.3, 2.5))  # single-column figure
plt.figure(figsize=(6.9, 2.5))  # double-column figure
```

When saving figures, use pdf format and `bbox_inches="tight"` to avoid clipping labels:

```python
plt.savefig("figure.pdf", bbox_inches="tight")
```

In case some plots need rasterization, use `dpi=400`:

```python
plt.savefig("figure.pdf", dpi=400, bbox_inches="tight")
```
```