# OmicsFold

**Reproducible binary classification for omics data, with training-only preprocessing and auditable held-out results.**

OmicsFold helps researchers ask whether a table of molecular measurements can
predict a two-class outcome on unseen samples. It aligns feature matrices with
sample metadata, compares classifiers against a prevalence baseline, and produces
an offline HTML report alongside machine-readable predictions, metrics, and split
assignments. Python, the command line, and R share the same Python engine.

Use it for count, expression, or other numeric feature tables when you need to
account for patients, cohorts, or collection time during evaluation. OmicsFold is
research software; feature rankings are exploratory and are not validated
biomarkers, significance tests, diagnoses, or clinical recommendations.

## What it does

- Matches samples by exact ID and validates inputs before fitting.
- Fits learned preprocessing only on training samples in every outer and inner fold.
- Supports stratified, grouped, cohort-holdout, and forward chronological evaluation.
- Compares a training-prevalence baseline with L2 logistic regression; optionally
  evaluates random forest and Gaussian naive Bayes or selects models through nested CV.
- Records held-out probabilities, split membership, preprocessing states, settings,
  checksums, and software versions for inspection and verified replay.
- Writes a self-contained HTML report that opens locally without a server.

## Install and try it

Requires **Python 3.10 or newer**. Until the package is published on PyPI, install
from a downloaded or cloned copy of this repository:

```sh
python -m pip install .
omicsfold example --output first-run
```

Run the installation command from the repository root. A virtual environment is
recommended. If the console command is unavailable, use `python -m omicsfold`.
After the PyPI release, installation will be `python -m pip install omicsfold`.

Open **`first-run/report.html`** in your browser. This example uses eight bundled
synthetic samples, requires no downloads, and demonstrates the output format.
Its scores are not evidence of biological performance. Each run requires a new
output directory; existing destinations are rejected.

## Prepare your data

Supply a CSV or TSV feature matrix and a separate metadata table. For example,
with samples in rows, the feature matrix is:

```csv
sample_id,gene_A,gene_B
001,10,3
002,2,12
```

The metadata table is:

```csv
sample_id,outcome,patient,cohort
002,control,P2,external
001,case,P1,development
```

These two rows illustrate the format only. Actual evaluation requires both classes
in every training and test partition, with enough independent samples or groups
for the requested folds. Metadata order may differ: OmicsFold aligns rows by ID.
IDs such as `001` remain strings. Duplicate IDs, duplicate feature names, unmatched
samples, missing outcomes, and invalid numeric values produce errors.

Keep outcomes and outcome proxies out of the feature matrix. Metadata are never
used as model features. Declare the outcome label whose probability you want to
predict with `--positive-label`.

| Input type | Default preprocessing |
| --- | --- |
| `counts` | Nonnegative integer counts with nonzero library sizes; sample-wise CPM, log1p, training expression filtering, constant removal, and scaling |
| `expression` or `numeric` | Training-all-missing feature removal, training median imputation, constant removal, and scaling |

Numeric missing tokens are empty cells, `NA`, `NaN`, `nan`, and `N/A`. Counts reject
missing values by default. Count filtering retains features with CPM >= 1 in at
least ceil(10% of training samples). Test data use the training feature mask,
imputer, and scaler. No usable training features is an error.

For features in rows, use `--orientation features-by-samples`: sample IDs become
column headers, and `feature_id` supplies feature names (change with
`--feature-id-column`). CSV and TSV inputs may be mixed.

## Run a benchmark

Validate your files first:

```sh
omicsfold validate --features features.tsv --metadata metadata.csv --orientation samples-by-features --sample-id-column sample_id --outcome-column outcome --positive-label case --input-type counts
```

Run a complete benchmark with the repository's synthetic fixture:

```sh
omicsfold benchmark --features examples/benchmark/counts.tsv --metadata examples/benchmark/metadata.csv --orientation samples-by-features --sample-id-column sample_id --outcome-column outcome --positive-label case --input-type counts --group-column patient --cohort-column cohort --split cohort --holdout-cohort external --seed 42 --output benchmark-results
```

Replace the paths and column names with your own. Choose a split that matches the
intended use of the model:

| Strategy | Use and options |
| --- | --- |
| Stratified CV | Independent samples; `--split stratified --folds 5` |
| Grouped CV | Repeated patients or related samples; `--split grouped --group-column patient` |
| Cohort holdout | A predeclared external cohort; `--split cohort --cohort-column cohort --holdout-cohort external` |
| Temporal | Future samples; `--split temporal --time-column DATE`; optionally `--time-cutoff ISO_DATE` and `--temporal-gap-days` |

Without `--split`, a named cohort holdout selects cohort mode; otherwise a group
column selects grouped CV; otherwise stratified CV is used. Supplied groups are
checked in every strategy, including cohort and temporal holdouts. Training and
test IDs and groups must be disjoint, and both sides must contain both classes.
A cohort column alone does not constrain CV membership.

The default classifier is L2 logistic regression (C=1, liblinear, up to 2,000
iterations), with seed 42. Add `--models logistic_l2 random_forest gaussian_nb`
for multiple families and `--tune` for training-only nested model selection.
See [model validation](docs/model-validation.md) and the runnable
[temporal example](examples/nested_temporal.py).

## Understand the results

Every successful benchmark creates six files:

| File | What to use it for |
| --- | --- |
| `report.html` | Review sample alignment, class balance, fold sizes, model comparisons, and precision-recall curves |
| `predictions.tsv` | Inspect held-out positive-class probabilities by sample, model, and fold |
| `metrics.tsv` | Compare average precision (AP), ROC AUC, prevalence, and test sizes per fold and as fold means |
| `splits.tsv` | Audit every training and test assignment, including groups and cohorts |
| `features.tsv` | Inspect retained features and exploratory model coefficients or importances |
| `run.json` | Recover configuration, provenance, fitted states, uncertainty estimates, versions, and checksums |

**Start with the prevalence baseline and the split design.** The baseline predicts
the training positive prevalence for every test sample. AP summarizes positive-class
ranking and depends on class balance; compare it with the baseline on the same test
partition. ROC AUC describes ranking across both classes; 0.5 corresponds to chance
ranking. Neither metric establishes probability calibration, clinical utility,
causation, or generalization to a different population.

Mean rows are arithmetic means of fold metrics, rather than scores from pooled
predictions. The report plots a named holdout or explicitly labeled fold, rather
than an unlabeled pooled CV curve. Temporal CV can leave initial training and
embargo samples unscored.

With a group column, the default gives each group total weight one in fitting and
evaluation, and uses 200 whole-group bootstrap resamples. Results still describe
sample predictions, with equal group contribution. Use `--weighting sample
--bootstrap-repeats 0` to request unweighted sample metrics without intervals.
Bootstrap intervals condition on the fitted training model; they do not measure
uncertainty from retraining. Small or dependent datasets can give unstable scores.
No particular level of performance is guaranteed.

Feature coefficients describe transformed, standardized inputs and the chosen
positive class. They are model-specific exploratory associations. Check stability
and confirm findings independently before drawing biological conclusions.
See the [full output schema](docs/report-schema.md).

## Use Python

```python
from omicsfold import load_data, benchmark

data = load_data(
    "examples/benchmark/counts.tsv", "examples/benchmark/metadata.csv",
    orientation="samples-by-features", sample_id_column="sample_id",
    outcome_column="outcome", positive_label="case", input_type="counts",
    group_column="patient", cohort_column="cohort",
)
result = benchmark(data, split="cohort", holdout_cohort="external", seed=42)
result.write("python-results")
```

The API and CLI use the same benchmark implementation. The separate `analyze()`
API is an exploratory workflow with different outputs and preprocessing; use
`benchmark()` for the evaluation protocol described here.

## Use R

The R package calls the installed Python engine and returns R data frames for
metrics, predictions, and splits, plus report paths and provenance. It requires
R packages `jsonlite` and `processx`, and a Python environment containing OmicsFold.
From the repository root, after installing the Python package:

```sh
Rscript -e "install.packages(c('jsonlite', 'processx'), repos='https://cloud.r-project.org')"
R CMD build r/omicsfold
R CMD INSTALL omicsfold_0.1.0.tar.gz
Rscript examples/benchmark.R
```

Set `OMICSFOLD_PYTHON` to the environment's Python executable if needed.
See the [R user guide](r/omicsfold/README.md) for an interactive example and
`?omicsfold_benchmark` for argument documentation. CRAN installation will be
available after the R package is accepted and published.

## Methods and examples

- [Research controls](docs/research-controls.md): assay-specific normalization,
  batch adjustment, metadata mapping, group weighting, uncertainty, and replay.
- [Model validation](docs/model-validation.md): nested selection, chronological
  splits, and safeguards against selection bias.
- [Public-data case study](examples/public_data/README.md): reproducible evaluation
  of historical ClinVar classification disagreement, with a
  [saved offline report](examples/public_data/published/report.html). This outcome
  is recorded disagreement, not pathogenicity. The saved report uses its documented
  historical sample-weighted protocol, rather than the current grouped defaults.

Assay transforms must suit the measurements. Batch adjustment rejects unseen test
batches and confounded training designs. Exact replay verifies inputs, fitted
states, results, and the measured environment; numerical identity across arbitrary
platforms or dependency versions is not assumed. Revising an analysis after viewing
held-out results requires independent confirmation.

## License and support

OmicsFold is available under the [MIT license](LICENSE).
Report bugs and request features through [GitHub Issues](https://github.com/ShahrozRahman/OmicsFold/issues).

