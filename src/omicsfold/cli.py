"""CSV/TSV validation and analysis adapters for the shared engine."""
import argparse
import csv
import json
import sys
from pathlib import Path
from importlib.resources import as_file, files
import pandas as pd
from .engine import analyze
from .io import load_data
from .benchmark import benchmark
from .reproducibility import replay
from .bundle import bundle_run


def _input_arguments(parser):
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--orientation", required=True, choices=["samples-by-features", "features-by-samples"])
    parser.add_argument("--sample-id-column", required=True)
    parser.add_argument("--outcome-column", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--input-type", required=True, choices=["counts", "expression", "numeric"])
    parser.add_argument("--group-column")
    parser.add_argument("--cohort-column")
    parser.add_argument("--feature-id-column", default="feature_id")


def read_csv(path, metadata=False):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV columns")
    return pd.read_csv(path, dtype=str if metadata else {"sample_id": str},
                       keep_default_na=False,
                       na_values=None if metadata else {c: ["", "NA", "NaN", "nan"] for c in header if c != "sample_id"})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="OmicsFold reproducible binary classification")
    commands = parser.add_subparsers(dest="command", required=True)
    replaying = commands.add_parser("replay", help="Verify environment, inputs and exact results against run.json")
    bundling = commands.add_parser("bundle", help="Preserve original inputs, source, results and dependency pins in a new ZIP")
    for command in (replaying, bundling):
        for flag in ("run", "features", "metadata", "output"):
            command.add_argument("--" + flag, required=True)
    example = commands.add_parser("example", help="Run the bundled synthetic cohort holdout offline")
    example.add_argument("--output", required=True)
    example.add_argument("--seed", type=int, default=42)
    validation = commands.add_parser("validate", help="Validate CSV/TSV inputs without fitting a model")
    analysis = commands.add_parser("analyze", help="Validate inputs and run cross-validation")
    benchmarking = commands.add_parser("benchmark", help="Compare prevalence with fixed or nested-selected models on held-out samples")
    for command in (validation, analysis, benchmarking):
        _input_arguments(command)
    validation.add_argument("--allow-missing-counts", action="store_true")
    for command in (validation, benchmarking):
        command.add_argument("--metadata-map", help="JSON file with explicit columns/values harmonization maps")
    benchmarking.add_argument("--output", required=True)
    benchmarking.add_argument("--split", choices=["stratified", "grouped", "cohort", "temporal"])
    benchmarking.add_argument("--holdout-cohort")
    benchmarking.add_argument("--folds", type=int, default=5)
    benchmarking.add_argument("--seed", type=int, default=42)
    benchmarking.add_argument("--dataset-label", default="Dataset")
    benchmarking.add_argument("--normalization", choices=["auto", "none", "cpm_log1p", "median_ratio_log1p", "log1p", "clr"], default="auto")
    benchmarking.add_argument("--missing-counts", choices=["error", "median"], default="error")
    benchmarking.add_argument("--batch-column")
    benchmarking.add_argument("--weighting", choices=["auto", "sample", "group_balanced"], default="auto")
    benchmarking.add_argument("--bootstrap-repeats", type=int)
    benchmarking.add_argument("--environment-lock", help="Prior run.json or environment JSON; mismatches are errors")
    benchmarking.add_argument("--bundle", help="Also preserve raw inputs, exact engine and runtime pins in a new ZIP")
    benchmarking.add_argument("--models", nargs="+", choices=["logistic_l2", "random_forest", "gaussian_nb"], default=["logistic_l2"])
    benchmarking.add_argument("--tune", action="store_true", help="Select candidates in inner folds only, inside every outer training partition")
    benchmarking.add_argument("--inner-folds", type=int, default=3)
    benchmarking.add_argument("--search-space", help="JSON candidate list; requires --tune")
    benchmarking.add_argument("--time-column")
    benchmarking.add_argument("--time-cutoff")
    benchmarking.add_argument("--temporal-gap-days", type=float, default=0.0)
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--folds", type=int, default=5)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--top-k", type=int)
    analysis.add_argument("--additional-group-column", action="append", default=[], help=argparse.SUPPRESS)
    # Keep the earlier flag-only analysis invocation, with the same explicit
    # input contract as the named analyze subcommand.
    if argv and argv[0].startswith("--") and argv[0] not in ("--help",):
        argv.insert(0, "analyze")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command in ("replay", "bundle"):
        try:
            function = replay if command == "replay" else bundle_run
            function(args["run"], args["features"], args["metadata"], args["output"])
        except (ValueError, OSError, KeyError, csv.Error) as exc:
            parser.exit(2, f"omicsfold {command}: {exc}\n")
        print("Exact environment, input and result checksums verified" if command == "replay" else "Reproducibility bundle written; original inputs included")
        return 0
    if command == "example":
        try:
            resources = files("omicsfold") / "data"
            with as_file(resources / "counts.tsv") as x, as_file(resources / "metadata.csv") as m:
                data = load_data(x, m, orientation="samples-by-features", sample_id_column="sample_id",
                                 outcome_column="outcome", positive_label="case", input_type="counts",
                                 group_column="patient", cohort_column="cohort")
                benchmark(data, split="cohort", holdout_cohort="external", seed=args["seed"],
                          dataset_label="Synthetic count fixture (8 samples)").write(args["output"])
        except (ValueError, OSError, csv.Error) as exc:
            parser.exit(2, f"omicsfold: {exc}\n")
        print(f"Report: {args['output']}/report.html")
        return 0
    output = args.pop("output", None)
    bundle_path = args.pop("bundle", None)
    selection_options = {key: args.pop(key) for key in ("models", "tune", "time_column", "time_cutoff", "temporal_gap_days") if key in args}
    if "inner_folds" in args:
        selection_options["inner_splits"] = args.pop("inner_folds")
    search_path = args.pop("search_space", None)
    additional_groups = args.pop("additional_group_column", [])
    benchmark_options = {key: args.pop(key) for key in ("split", "holdout_cohort", "dataset_label", "normalization",
                         "missing_counts", "batch_column", "weighting", "bootstrap_repeats", "environment_lock") if key in args}
    mapping_path = args.pop("metadata_map", None)
    analysis_options = {key: args.pop(key) for key in ("folds", "seed", "top_k") if key in args}
    features, metadata = args.pop("features"), args.pop("metadata")
    try:
        if mapping_path:
            with open(mapping_path, encoding="utf-8") as handle:
                args["metadata_mapping"] = json.load(handle)
        if search_path:
            with open(search_path, encoding="utf-8") as handle:
                selection_options["search_space"] = json.load(handle)
        if benchmark_options.get("missing_counts") == "median":
            args["allow_missing_counts"] = True
        data = load_data(features, metadata, **args)
        if command == "validate":
            print(json.dumps(data.summary(), indent=2))
        elif command == "benchmark":
            result = benchmark(data, **benchmark_options, **selection_options, n_splits=analysis_options["folds"],
                               seed=analysis_options["seed"])
            result.provenance["orientation"] = args["orientation"]
            result.write(output)
            if bundle_path:
                bundle_run(Path(output) / "run.json", features, metadata, bundle_path)
        else:
            result = analyze(data.features, data.metadata, positive_label=data.positive_label,
                             group_columns=[c for c in (data.group_column, data.cohort_column) if c] + additional_groups,
                             n_splits=analysis_options["folds"], seed=analysis_options["seed"],
                             top_k=analysis_options["top_k"])
            result.provenance["input_type"] = data.input_type
            result.provenance["orientation"] = args["orientation"]
            result.write(output)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(2, f"omicsfold: {exc}\n")
    return 0
