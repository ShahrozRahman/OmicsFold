"""Equal group contribution and whole-cluster resampling."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights


def cluster_intervals(y, scores, groups, weights, repeats, seed):
    if not repeats:
        return None
    clusters = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if len(clusters) < 2:
        raise ValueError("Cluster intervals require at least two test groups")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        if len(np.unique(y[indices])) != 2:
            continue
        values.append([average_precision_score(y[indices], scores[indices], sample_weight=weights[indices]),
                       roc_auc_score(y[indices], scores[indices], sample_weight=weights[indices])])
    if len(values) < max(20, repeats // 2):
        raise ValueError("Too few two-class cluster bootstrap resamples for reliable intervals")
    bounds = np.quantile(values, [.025, .975], axis=0)
    return {"method": "whole-test-group percentile bootstrap; conditional on the fitted training model",
            "confidence": .95, "requested_resamples": repeats, "valid_resamples": len(values),
            "average_precision": bounds[:, 0].tolist(), "roc_auc": bounds[:, 1].tolist()}
