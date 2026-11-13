# OmicsFold for R

This package calls the installed Python engine. It does not duplicate splitting,
preprocessing, fitting, metrics or HTML rendering in R. R dependencies are
`jsonlite` and `processx`; Python requires the installed `omicsfold` package.

From the repository root, install the Python engine into your chosen Python environment:

```sh
python -m pip install .
```

Set `OMICSFOLD_PYTHON` to that environment's Python executable before starting
R, or pass its path as `executable`. The package does not install Python or
Python dependencies automatically.

To build the R package from the repository root:

```sh
R CMD build r/omicsfold
R CMD INSTALL omicsfold_0.1.0.tar.gz
Rscript examples/benchmark.R
```

In R:

```r
library(omicsfold)
x <- read.delim(system.file("extdata", "counts.tsv", package = "omicsfold"),
                colClasses = c(sample_id = "character"), check.names = FALSE)
m <- read.csv(system.file("extdata", "metadata.csv", package = "omicsfold"),
              colClasses = "character", check.names = FALSE)
result <- omicsfold_benchmark(
  x, m, positive_label = "case", input_type = "counts",
  output = "new R output directory",
  orientation = "samples-by-features",
  sample_id_column = "sample_id", outcome_column = "outcome",
  group_column = "patient", cohort_column = "cohort",
  split = "cohort", holdout_cohort = "external", seed = 42L,
  executable = Sys.getenv("OMICSFOLD_PYTHON", "python"))
result$metrics
result$predictions
result$splits
result$report
```

Set `executable` to the full Python path for the environment containing the
installed package (paths containing spaces are supported). With
`executable_type="python"`, the argument vector is
`-m omicsfold benchmark ...`. To use an installed console script instead, set
`executable="/path/to/omicsfold"` and `executable_type="cli"`. Supply just the
executable name/path, without embedded arguments or extra quoting. processx
launches it directly, without a shell, so labels and paths are literal arguments.
Preflight errors explain how to install Python OmicsFold in the selected
environment; later engine validation errors retain the CLI diagnostic.

The CLI contract passes explicit `--features`, `--metadata`, `--orientation`,
`--sample-id-column`, `--outcome-column`, `--positive-label`, `--input-type`,
`--output`, `--feature-id-column`, `--folds`, `--seed` and `--dataset-label`.
Optional grouping/cohort/split/holdout arguments map to their corresponding CLI
flags. NULL split uses the same Python default: cohort if a holdout is supplied,
otherwise grouped if a group column exists, otherwise stratified CV.

Matrices use explicit dimnames: sample rows and feature columns for
`samples-by-features`, or feature rows and sample columns for
`features-by-samples`. Data frames can use the declared ID column instead;
otherwise explicit row names are required (automatic row numbers are rejected).
Keep sample IDs as character strings, including `001` and literal `NA`.
Metadata are separate and matched by exact ID in Python. Factor metadata are
serialized as their labels, not integer codes. Real numeric features are written
with 17 significant digits; missing feature cells are empty. Original names are
not repaired, and UTF-8, spaces, punctuation and quotes are preserved. Existing
output paths are rejected and input objects are not modified.

Results use [benchmark report schema 1.0](../../docs/report-schema.md): exactly
six files in a new output directory. Return values are data frames `metrics`
(including fold means), `predictions` (held-out probabilities), and `splits`
(both train and test membership), plus absolute paths `report`, `output` and
parsed provenance `run`. IDs/labels remain character columns, metric/fold values
are numeric, and blank mean-row fold values become NA. Metrics are sample-level
with a user-specified positive label. Exploratory retention/coefficients remain
available in `features.tsv`, with no biological significance claims.

The bundled `inst/extdata/` fixture contains eight synthetic samples and makes
no biological performance claims.

Additional controls pass directly to Python: `normalization`, `missing_counts`,
`batch_column`, `weighting`, `bootstrap_repeats`, `metadata_map`, and
`environment_lock`, and `bundle`. Grouped runs default to equal group contribution and
200 whole-group resamples. Select `weighting="sample", bootstrap_repeats=0L`
explicitly for sample weighting. Fitted states and intervals are recorded in
`result$run$folds`. See `docs/research-controls.md` for assumptions and replay.

Nested model selection and chronological validation use `models`, `tune`,
`inner_folds`, `search_space` (JSON path), `time_column`, `time_cutoff` and
`temporal_gap_days`. Every inner fit receives only outer training samples and
refits preprocessing within its inner training fold. Date/POSIXt metadata retain
temporal order (POSIXt is converted to UTC). The run record includes selected
candidates, inner memberships/scores and chronological bounds. See
`docs/model-validation.md` for the protocol and its assumptions.
