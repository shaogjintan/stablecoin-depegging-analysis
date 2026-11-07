"""Unused historical AP wrapper retained for reference."""
from sklearn.metrics import average_precision_score

def _average_precision(y_true, scores) -> float:
    """Average precision, matching the metric used in final_results."""
    return float(average_precision_score(y_true, scores))

