"""Unused historical L1/L2 search grid; final protocol tunes C for L1 only."""

PARAM_GRID = [
        {"C": c, "penalty": p}
        for p in ("l1", "l2")
        for c in (0.001, 0.01, 0.1, 1.0)
    ]
