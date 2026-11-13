"""Run installed CLI nested model selection on synthetic future samples."""
import argparse
import json
from pathlib import Path
import subprocess
import sysconfig
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/results/nested-temporal"))
    args = parser.parse_args()
    work = args.output.resolve()
    if work.exists():
        raise FileExistsError(f"Example work directory already exists: {work}")
    rng = np.random.default_rng(19)
    n = 60
    y = np.tile([1, 0], n // 2)
    values = rng.normal(size=(n, 5))
    values[:, 0] += y * 1.2
    x = pd.DataFrame(values, columns=[f"synthetic_feature_{i}" for i in range(5)])
    x.insert(0, "sample_id", [f"S{i:03}" for i in range(n)])
    m = pd.DataFrame({"sample_id": x.sample_id, "outcome": np.where(y, "case", "control"),
                      "patient": [f"synthetic_pair_{i//2:03}" for i in range(n)],
                      "collected_at": [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i//2)).date().isoformat() for i in range(n)]})
    grid = [{"model": "logistic_l2", "C": .1}, {"model": "logistic_l2", "C": 1.},
            {"model": "random_forest", "n_estimators": 32, "max_depth": 3, "min_samples_leaf": 2},
            {"model": "gaussian_nb", "var_smoothing": 1e-9}]
    cli = Path(sysconfig.get_path("scripts")) / ("omicsfold.exe" if os.name == "nt" else "omicsfold")
    if not cli.is_file():
        raise RuntimeError("Install this package first; this example exercises the installed console script")
    work.mkdir(parents=True)
    x.to_csv(work / "features.tsv", sep="\t", index=False)
    m.to_csv(work / "metadata.tsv", sep="\t", index=False)
    (work / "search.json").write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    command = [str(cli), "benchmark", "--features", "features.tsv", "--metadata", "metadata.tsv",
               "--orientation", "samples-by-features", "--sample-id-column", "sample_id", "--outcome-column", "outcome",
               "--positive-label", "case", "--input-type", "numeric", "--group-column", "patient", "--split", "temporal",
               "--time-column", "collected_at", "--time-cutoff", "2020-01-25", "--temporal-gap-days", "1", "--tune",
               "--models", "logistic_l2", "random_forest", "gaussian_nb", "--inner-folds", "3", "--search-space", "search.json",
               "--bootstrap-repeats", "60", "--seed", "42", "--dataset-label", "Synthetic nested temporal holdout (60 samples)",
               "--output", "results"]
    subprocess.run(command, cwd=work, check=True)
    (work / "command.json").write_text(json.dumps(["omicsfold", *command[1:]], indent=2) + "\n")
    print(f"Synthetic report: {work / 'results/report.html'}")


if __name__ == "__main__":
    main()
