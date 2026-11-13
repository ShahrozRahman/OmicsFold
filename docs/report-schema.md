# Benchmark report schema 1.0

`BenchmarkResult.write(directory)` and `omicsfold benchmark --output directory`
use the same writer. A successful run contains exactly six UTF-8 files. It never
overwrites a destination, whether a file, directory or symlink. Files are staged
before publication; rendering failures leave no result directory. No reports are
produced when input/split/preprocessing validation fails. This schema applies to
`benchmark`; the earlier exploratory `analyze` artifacts are separate.

TSV tables have a header row, tabs as separators, standard CSV quoting, and empty
fields for unavailable values. Read `sample_id` as text to preserve leading zeros.
Folds are zero-based. Class labels, IDs, feature names, cohorts and groups are
literal user strings. `model` is `prevalence`, a fixed family (`logistic_l2`,
`random_forest`, `gaussian_nb`), or `selected_model` for a nested selection policy.

| File | Columns |
| --- | --- |
| predictions.tsv | sample_id, outcome, positive_label, fold, model, positive_probability, evaluation_weight |
| metrics.tsv | scope, fold, model, average_precision, roc_auc, test_positive_prevalence, test_sample_count |
| splits.tsv | fold, partition, sample_id, outcome, group, cohort |
| features.tsv | fold, model, feature, retained, reason, coefficient, absolute_coefficient_rank, interpretation |

`predictions.tsv` contains only held-out samples. In CV, each sample occurs once
per model; in cohort mode, only the named cohort receives scores.
Forward temporal CV scores only future test blocks; its initial training block
and embargo rows may be unscored. See `validation_design` for coverage.
`evaluation_weight` is one per sample in sample mode and 1/group-size in the
new default group-balanced mode. Metrics and PR curves use these weights.
The six-file schema remains compatible; run.json additionally records per-fold
cluster intervals, fitted preprocessing states, environment locks and exact-result
checksums. See research-controls.md for statistical and replay semantics.

`metrics.tsv` includes one `scope=fold` row per model/fold, followed by one
`scope=mean` row per model. Mean rows have an empty fold field. Every numeric
column is averaged arithmetically across folds without sample weighting; thus
mean `test_sample_count` can be noninteger. ROC AUC and average precision always
use the user-specified positive label. A mean is not a metric computed from
pooled predictions. With a single holdout, means equal the fold results.

`splits.tsv` records both `partition=train` and `partition=test`. A CV sample can
appear as training in multiple folds; this is expected. No ID/group may appear
on both sides of the same fold. Cohort/group fields are empty if not supplied.

`features.tsv` has one row per input feature/fold for the fitted model. `retained` is
True or False. Reasons are `retained`, `expression_filter`,
`all_missing_in_training`, or `constant_in_training`. Coefficients describe
standardized transformed training features for the positive label. Retained
features receive a rank starting at 1 by descending absolute coefficient;
ties are ordered by feature name. Dropped features have empty coefficient and
rank fields. The prevalence baseline has no feature coefficients. Rankings are
exploratory model outputs, not validated biomarkers or inferential statistics.
Non-logistic families have empty coefficient/rank fields. Random-forest folds
add `impurity_importance`; Gaussian naive Bayes supplies retention only.

`run.json` contains:

Nested/temporal extensions include `validation_design` (candidate policy and
scored/unscored IDs) and per-fold `selected_candidate`, actual `model_parameters`,
`selection` (inner memberships, preprocessing states and candidate scores), and
`temporal` (date boundaries and IDs outside that fold). Actual per-fold parameters
are authoritative; the legacy `logistic_regression` block describes untuned defaults.

- `report_schema_version`: `"1.0"`; `output_files`: the six filenames.
- `dataset_label`, `positive_label`, `seed`, `split`, `n_splits`, and
  `holdout_cohort`; resolved loading/analysis settings under `configuration`.
- `input_checksums.source_files`: per-role (`features`, `metadata`) original
  path, SHA-256 of exact bytes read, and size in bytes. Empty for inputs created
  directly in memory. `canonical_features_sha256` and `canonical_metadata_sha256`
  hash UTF-8 CSV serialization (no index, `%.17g` floats) of the actual aligned
  benchmark input tables, sorted by sample ID. Canonical hashes are separate
  from raw-file hashes and include original feature/metadata column order.
- `package_versions`: Python and scientific package versions.
- `sample_alignment_audit.source_alignment`: source sample positions, matched
  count, metadata reorder count, and unmatched/duplicate counts. Positions are
  zero-based sample positions in the declared orientation. Failed matches are
  rejected before reporting; successful unmatched/duplicate counts are zero.
  For direct in-memory inputs this describes the supplied frames.
- `sample_alignment_audit.benchmark_alignment`: alignment of the frames passed
  to the engine; `analysis_order` records sorting by sample ID.
- `class_balance`: input outcome counts.
- `folds`: train/test IDs, training prevalence, retained features and logistic
  intercept (standardized feature space) for each fold.
- `metric_summary`: aggregation definition and all fold/mean metric rows.

Additional provenance keys document preprocessing and model settings. Preserve
input files, source revision and environment alongside this record for future
reproduction; hashes and version strings alone are not copies of those inputs.

`report.html` is standalone, uses inline CSS/SVG, and loads no external assets.
It escapes user text and shows source alignment, class balance, fold sizes,
per-fold and mean metrics, and exploratory feature outputs. Rankings display the
top 20 retained features per fold; the TSV includes every feature. Its PR curve
uses only the named cohort holdout, or clearly labeled fold 0 for CV. Results
are sample-level; there are no patient-level aggregates or uncertainty intervals.
