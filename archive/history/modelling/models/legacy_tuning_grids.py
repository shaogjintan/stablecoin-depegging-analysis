"""Unused model search grids retained for historical reference.

The maintained protocol uses fixed RF/XGB settings and training-only LR C selection.
"""

BASEMODEL_PARAM_GRID = [{}]

RFMODEL_PARAM_GRID = [
        {"max_depth": d, "min_samples_leaf": leaf, "class_weight": cw}
        for d in (6, 12, None)
        for leaf in (5, 20, 50)
        for cw in (None, "balanced_subsample")
    ]

XGBMODEL_PARAM_GRID = [
        {"learning_rate": lr, "max_depth": d, "min_child_weight": m,
         "scale_pos_weight": s}
        for lr in (0.03, 0.1)
        for d in (3, 5)
        for m in (1, 5)
        for s in ("auto", 10, 100, 500)  # 500 added for the recall-first rework
    ]

CONSTANTPRIORMODEL_PARAM_GRID = [{}]

PERSISTENCEMODEL_PARAM_GRID = [{}]
