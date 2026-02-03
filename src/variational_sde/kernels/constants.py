from __future__ import annotations

from typing import Final

import triton.language as tl

GATE_R: tl.constexpr = tl.constexpr(0)
GATE_Z: tl.constexpr = tl.constexpr(1)
GATE_N: tl.constexpr = tl.constexpr(2)
NUM_GATES: tl.constexpr = tl.constexpr(3)
NUM_GATES_INT: Final[int] = 3

MAX_LAYERS: Final[int] = 4
