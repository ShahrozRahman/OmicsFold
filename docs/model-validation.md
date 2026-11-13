# Model selection and validation

Python, the CLI and R support fixed model families, nested tuning, chronological
holdout/forward CV, and cohort holdout. Defaults retain fixed logistic regression
for small fixtures and backward compatibility; it is no longer the only model.
The six-file output structure is unchanged.

## Model families

- `logistic_l2`: liblinear L2, C=1 by default, tolerance 1e-4, maximum 2,000
  iterations. The default tuning grid uses C=0.01, 0.1, 1 and 10.
- `random_forest`: 200 trees, minimum leaf size 2, fixed seed and one worker.
  Default tuning compares maximum depth 3 and unrestricted depth. Other settings
  retain the recorded sklearn defaults. Impurity importances are exploratory,
  can favor correlated/high-cardinality inputs, and are not coefficients.
- `gaussian_nb`: Gaussian naive Bayes; default var_smoothing=1e-9. The default
  tuning grid compares 1e-9 and 1e-7. No single coefficient ranking is reported.

Select a predeclared fixed family with `--models gaussian_nb`, for example.
Multiple candidates require `--tune`, which separates selection from evaluation:

```sh
omicsfold benchmark ... --models logistic_l2 random_forest gaussian_nb --tune --inner-folds 3 --output new-results
```

Replace `...` with the explicit input declarations from README. A custom
`--search-space candidates.json` accepts a list such as:

```json
[
  {"model": "logistic_l2", "C": 0.1},
  {"model": "logistic_l2", "C": 1.0},
  {"model": "random_forest", "n_estimators": 200, "max_depth": 3, "min_samples_leaf": 2},
  {"model": "gaussian_nb", "var_smoothing": 1e-9}
]
```

Unsupported/invalid parameters fail rather than being skipped. Declare the
search space and budget before viewing outer scores. Equal inner scores choose
the first candidate in the recorded order.

## Nested evaluation

For every outer fold:

1. Split **outer training data only** into inner folds. Use stratification for
   independent data, stratified group CV for supplied groups, and forward
   chronological validation for temporal designs.
2. Refit learned preprocessing inside each inner training fold: imputation,
   reference normalization, filtering, batch adjustment and scaling. The fixed
   preprocessing configuration is shared across candidates within that fold;
   this search tunes models, not preprocessing choices.
3. Select by mean inner-fold AP, applying the selected sample/group weighting
   within each fold. Validation scores are never taken from the outer test set.
4. Refit preprocessing and the selected model on all outer training samples;
   score the untouched outer test partition alongside training prevalence.

The outer output `selected_model` evaluates the **selection procedure**, which
can choose different families/parameters in different outer folds. It does not
publish unselected candidates' outer scores for picking a winner afterward.
Fixed-family outputs retain their model names. Inner IDs, fitted states, trial
scores and choices are recorded under `run.json -> folds -> selection`. The HTML
shows per-outer-fold choices and inner trial scores. All existing class/ID/group
checks apply inside and outside; insufficient or invalid inner folds fail.

This follows the separation described in
[scikit-learn's nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Nested CV protects the declared search. Repeatedly changing features, candidate
grids or protocols based on outer scores still reuses evaluation data and can
introduce selection bias; an independently reserved final dataset is needed.

## Chronological validation

`--split temporal --time-column collected_at` uses expanding forward folds over
distinct timestamps. Equal timestamps stay together; future samples never enter
training. The initial block supplies training only and receives no held-out score.
Later test blocks are disjoint. Earlier test periods can join training in later
folds, as in rolling deployment; each current test period remains unseen when scored.

Add `--time-cutoff 2020-01-25` for a single predeclared holdout: training is
strictly before the cutoff and test is on/after it. `--temporal-gap-days 2`
requires training to end strictly before each test boundary minus two days.
Embargo/future rows outside a fold are recorded rather than silently scored.

Use ISO dates/timestamps; naive dates are interpreted as UTC and explicit zones
are converted to UTC. Missing/ambiguous dates and guessed numeric epochs are
rejected. R Date values serialize as ISO dates and POSIXt in UTC. Supplied
groups remain disjoint: a patient spanning a time boundary causes rejection,
not an automatic removal of patients. Inner temporal folds respect the same
ordering and embargo within their outer training period.

`validation_design` records scored and unscored IDs. Each temporal fold records
latest train time, earliest test time and all IDs outside that particular fold,
including embargo and future rows. Temporal CV need not score every input row.

## Uncertainty and external evidence

Existing whole-group bootstrap intervals remain available (`--bootstrap-repeats
200`, any value >=40). With groups they default to 200 resamples; without groups,
explicit bootstrap requests resample samples. They condition on the fitted or
selected training model, do not bootstrap the complete tuning/retraining process,
and are not averaged across folds. See research-controls.md for assumptions.

A named external cohort can remain completely held out while all tuning occurs
inside the other cohorts. A temporal holdout provides future-time evidence.
Neither is a general proof of validity for all populations, hospitals, assays or
future periods. Reports record the design and observed evidence without such a
claim. The real expression-cohort case remains pending its existing compatibility
checks; software features do not replace those scientific checks.

## Reproducible synthetic demonstration

```sh
python examples/nested_temporal.py
```

The installed CLI selects among four predeclared candidates using three inner
forward folds, applies a one-day embargo and scores future samples. Open
`examples/results/nested-temporal/results/report.html`. All 60 samples are
synthetic paired observations; this demonstrates a workflow, not biological
external validity. Use `--output NEW_WORK_DIRECTORY` to run again.

Python options: `models`, `tune`, `inner_splits`, `search_space` (candidate list),
`time_column`, `time_cutoff`, `temporal_gap_days`. R uses `inner_folds` and a JSON
path for `search_space`. Verified replay/bundles preserve these choices and states.
