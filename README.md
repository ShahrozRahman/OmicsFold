# OmicsFold

[![PyPI](https://img.shields.io/pypi/v/omicsfold)](https://pypi.org/project/omicsfold/)
[![Python](https://img.shields.io/pypi/pyversions/omicsfold)](https://pypi.org/project/omicsfold/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproducible binary classification for omics data, available through Python, the command line, and R.

OmicsFold compares classifiers on held-out samples and produces a self-contained HTML report with predictions, metrics, and an auditable record of each run. It accepts count, expression, and numeric feature tables with separate sample metadata.

- **Training-only preprocessing:** filtering, imputation, and scaling are fitted within each training fold.
- **Study-aware evaluation:** stratified and grouped cross-validation, external cohort holdout, and chronological splits.
- **Model comparison:** prevalence baseline, L2 logistic regression, random forest, and Gaussian naive Bayes, with optional nested model selection.
- **Reproducible outputs:** split assignments, fitted preprocessing states, input checksums, software versions, and verified replay.

## Installation

Requires **Python 3.10 or later**.

```sh
python -m pip install omicsfold
```

The Python package is available on [PyPI](https://pypi.org/project/omicsfold/). The R package has been submitted to CRAN and is awaiting review; source installation is described below.

## Quick start

Run the bundled example without downloading data:

```sh
omicsfold example --output first-run
```

Open `first-run/report.html` in your browser. The example uses eight synthetic samples to demonstrate the workflow and report format. Use a new output directory for each run.

If the console command is unavailable, use `python -m omicsfold`.

## Input data

Provide a CSV or TSV feature matrix and a separate metadata table. With samples in rows:

**features.tsv** (tab-separated)

```tsv
sample_id	gene_A	gene_B
001	10	3
002	2	12
```

**metadata.csv**

```csv
sample_id,outcome,patient,cohort
001,case,P1,development
002,control,P2,external
```

These rows illustrate the format; evaluation requires enough samples or independent groups, with both classes in every training and test partition.

Samples are matched by exact ID, regardless of metadata order. Keep IDs as strings and outcomes out of the feature matrix. Metadata are not used as model features. Duplicate IDs, unmatched samples, and invalid values are rejected.

| Input type | Default preprocessing |
| --- | --- |
| `counts` | CPM normalization, log1p, training expression filtering, constant removal, and scaling |
| `expression` or `numeric` | Training-all-missing feature removal, training median imputation, constant removal, and scaling |

Counts must be nonnegative integers with nonzero library sizes; missing counts are rejected by default. For features in rows, use `--orientation features-by-samples` with a `feature_id` column and sample IDs as column headers.

## Command line

Validate your inputs:

```sh
omicsfold validate --features features.tsv --metadata metadata.csv --orientation samples-by-features --sample-id-column sample_id --outcome-column outcome --positive-label case --input-type counts
```

Run grouped cross-validation, keeping each patient's samples together:

```sh
omicsfold benchmark --features features.tsv --metadata metadata.csv --orientation samples-by-features --sample-id-column sample_id --outcome-column outcome --positive-label case --input-type counts --split grouped --group-column patient --folds 5 --seed 42 --output results
```

Choose the evaluation strategy that matches your study:

| Strategy | Options |
| --- | --- |
| Independent samples | `--split stratified --folds 5` |
| Repeated patients or related samples | `--split grouped --group-column patient --folds 5` |
| External cohort | `--split cohort --cohort-column cohort --holdout-cohort external` |
| Future samples | `--split temporal --time-column collection_date` |

Supplied groups must remain disjoint between training and test sets, including cohort and temporal holdouts. A cohort column alone does not enforce cohort separation.

Logistic regression is the default classifier. Add `--models logistic_l2 random_forest gaussian_nb` to compare model families, or `--tune` to select candidates within training-only inner folds. Run `omicsfold benchmark --help` for all options.

## Python

```python
from omicsfold import benchmark, load_data

data = load_data(
    "features.tsv",
    "metadata.csv",
    orientation="samples-by-features",
    sample_id_column="sample_id",
    outcome_column="outcome",
    positive_label="case",
    input_type="counts",
    group_column="patient",
)
result = benchmark(data, split="grouped", n_splits=5, seed=42)
result.write("python-results")
```

The Python API and CLI use the same benchmark engine. Use `benchmark()` for this evaluation workflow; `analyze()` provides a separate exploratory workflow.

## R

The R package uses the installed Python engine and returns metrics, predictions, and split assignments as R data frames, alongside the HTML report and run provenance.

Until CRAN publication, install from a cloned or downloaded copy of this repository. First install the Python package as shown above, then run from the repository root:

```sh
Rscript -e "install.packages(c('jsonlite', 'processx'), repos='https://cloud.r-project.org')"
R CMD build r/omicsfold
R CMD INSTALL omicsfold_0.1.0.tar.gz
Rscript examples/benchmark.R
```

Set `OMICSFOLD_PYTHON` to your Python executable if needed. See the [R user guide](r/omicsfold/README.md) for an interactive example and `?omicsfold_benchmark` for arguments.

## Results

Each benchmark writes six files:

| File | Contents |
| --- | --- |
| `report.html` | Offline report with data checks, fold sizes, model comparisons, and precision-recall curves |
| `metrics.tsv` | Average precision, ROC AUC, prevalence, and test sizes per fold and as fold means |
| `predictions.tsv` | Held-out positive-class probabilities by sample, model, and fold |
| `splits.tsv` | Training and test assignments, including groups and cohorts |
| `features.tsv` | Retained features and exploratory coefficients or importances |
| `run.json` | Configuration, fitted states, uncertainty estimates, versions, and checksums |

Compare models with the prevalence baseline on the same test partitions. Fold means are arithmetic means, not scores from pooled predictions. Grouped runs default to equal group contribution and 200 whole-group bootstrap resamples; intervals condition on the fitted models and do not include retraining uncertainty.

OmicsFold is research software. Feature rankings are exploratory associations, and held-out scores do not establish clinical validity or generalization to a different population.

## Documentation and examples

- [Model validation](docs/model-validation.md): nested selection, chronological splits, and evaluation safeguards.
- [Research controls](docs/research-controls.md): normalization, batch adjustment, group weighting, uncertainty, and replay.
- [Output schema](docs/report-schema.md): report files and provenance fields.
- [Temporal example](examples/nested_temporal.py): chronological evaluation with nested selection.
- [Public-data case study](examples/public_data/README.md): historical ClinVar classification disagreement, with a [saved report](examples/public_data/published/report.html). The outcome is recorded disagreement, not pathogenicity; the report uses the documented historical sample-weighted protocol.

## Support and license

Report bugs or request features through [GitHub Issues](https://github.com/ShahrozRahman/OmicsFold/issues). Include the package version, command or code, and error message when reporting a problem.

Created by Shahroz Rahman. Released under the [MIT license](LICENSE).
