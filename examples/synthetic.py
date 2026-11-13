"""Synthetic demonstration only; no biological samples or claims."""
from pathlib import Path
import numpy as np
import pandas as pd
from omicsfold import analyze


def synthetic_data():
    rng = np.random.default_rng(17)
    n = 80
    outcome = np.tile(["control", "case"], n // 2)
    X = rng.normal(size=(n, 12))
    X[:, 0] += (outcome == "case") * 1.2
    X[0, 2] = np.nan
    features = pd.DataFrame(X, columns=[f"synthetic_feature_{i}" for i in range(12)])
    features.insert(0, "sample_id", [f"S{i:03}" for i in range(n)])
    metadata = pd.DataFrame({"sample_id": features.sample_id, "outcome": outcome,
                             "patient": [f"P{i // 2:03}" for i in range(n)]})
    return features, metadata


if __name__ == "__main__":
    features, metadata = synthetic_data()
    output = Path(__file__).parent / "results"
    result = analyze(features, metadata, positive_label="case", group_columns=["patient"], top_k=5)
    result.write(output)
    features.to_csv(output / "synthetic_features.csv", index=False)
    metadata.to_csv(output / "synthetic_metadata.csv", index=False)
    print(result.metrics)
