# Preprocessing, group inference and verified reproducibility

Python, the CLI and R share these controls. The new default for supplied groups
is **equal group contribution** and **200 whole-group bootstrap resamples**.
Earlier sample-weighted estimates are historical results, not the same estimand.

## Assay transformations

| `--normalization` | Method |
| --- | --- |
| auto | counts: CPM/log1p; expression/numeric: none |
| cpm_log1p | sample-wise CPM and natural log1p |
| median_ratio_log1p | RNA counts: median ratios to a frozen training geometric-mean reference, then log1p |
| log1p | natural log(x+1) for nonnegative abundance tables |
| clr | compositional abundance: log(x+1) minus each sample's mean log(x+1), fixed pseudocount 1 |
| none | suitable/preprocessed numerical data; training imputation and scaling still apply |

The RNA method uses genes positive in every training sample. Geometric reference
means and the geometric centering of training size factors are frozen. Each
test sample uses its own positive ratios against that training reference and
the frozen center. No reference genes or usable test ratios means an error.
This uses the frozen-reference principle described for
[DESeq2 size factors](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html),
not DESeq2 differential expression, VST, rlog, TMM or automatic assay detection.
Methods require assay-specific scientific justification.

`--missing-counts median` explicitly enables experimental training-median
imputation. Observed counts must still be finite nonnegative integers and each
sample must have an observed positive count. Medians are fitted inside training
folds; all-training-missing features are dropped. Imputed estimates may be
fractional and are not recovered measurements or a single-cell dropout model.
The default remains rejection of missing counts. In Python also load with
`allow_missing_counts=True`; the benchmark CLI does this for the median option.

All filtering, masks, references, medians, batch coefficients and scalers are
training-only. Fitted states are recorded in `run.json -> folds -> preprocessing`.

## Training-only batch adjustment

`--batch-column batch` fits an additive training regression with intercept,
training phenotype and batch indicators. It subtracts the centered batch
component while protecting the additive training phenotype association. Test
outcomes are never used. Every training batch must have both outcomes and the
design must have full rank. Test batches must be seen in training; unseen
batches are rejected rather than fitted using held-out observations.
This is additive location adjustment, **not ComBat**. Partial confounding,
interactions and nonlinear effects still require scientific review.

## Metadata harmonization

`--metadata-map maps.json` automatically applies an explicit schema/value map
before validation and records it in the alignment audit:

```json
{
  "columns": {"Subject": "sample_id", "Diagnosis": "outcome", "Center": "batch"},
  "values": {"outcome": {"affected": "case", "healthy": "control"}}
}
```

Roles refer to resulting column names. Mapped value columns require complete
coverage; unmapped values and column collisions fail. Sample ID values may not
be relabeled, trimmed or guessed. Phenotype equivalence is supplied by the
researcher, not inferred. Python accepts the object as `metadata_mapping` in
`load_data`; R accepts the JSON path as `metadata_map`.

## Equal group contribution and dependence-aware intervals

`--weighting auto` selects `group_balanced` with groups, otherwise `sample`.
A group with n samples gives each weight 1/n, for total weight one in both
training loss and evaluation. The prevalence baseline, test prevalence, AP,
AUC and PR curve all use these weights, saved as `evaluation_weight` in
predictions.tsv. Mixed-label gene groups are supported; no group outcome is
invented by averaging labels.

The bootstrap draws entire test groups with replacement, keeping their samples
together and retaining original within-group weights. Repeated groups contribute
repeatedly. One-class resamples are skipped; too few valid resamples cause an
error. At least two test groups are required. Set `--bootstrap-repeats N` with
N >=40, or explicitly select 0 to disable. The 95% percentile intervals are
conditional on the fitted training model, not retraining uncertainty. They are
stored per model/fold and are not averaged across folds.

These methods account for dependence within supplied clusters. They do not make
observations independent. Inference assumes groups are independent sampling
units; family/shared-donor links spanning nominal groups need an appropriate
cluster definition or study design. Equal weighting and independence are distinct.

## Exact replay: verification or refusal

Numerical libraries run with one thread. Each benchmark records Python,
OS/machine, dependency and numerical-library metadata, Python executable and
package implementation hashes, input checksums, preprocessing state and exact
hashes of predictions, metrics, membership, coefficients and fold states.

```sh
omicsfold replay --run previous/run.json --features features.tsv --metadata metadata.csv --output verified-replay
```

Replay restores the saved loading/analysis configuration, verifies the measured
environment and inputs, recomputes the analysis and publishes only if every
result hash matches. Mismatches leave no reproduced output. A prior run.json can
also be passed as `--environment-lock` on a new benchmark; that verifies only the
environment, whereas replay verifies inputs and results too. Older runs without
measured locks/checksums cannot be retroactively certified.

The measured lock is not an environment installer. Keep pinned environments or
containers and source inputs alongside it; a hash is not a copy. NumPy itself
[documents platform/build variation](https://numpy.org/doc/2.0/reference/random/compatibility.html).
The guarantee is **verified equality or a clear refusal**, not bit identity for
arbitrary hardware/dependency combinations. Unrecorded hardware behavior can
still differ; actual result hashes are the final gate.

## Complete example

Save original inputs automatically with `--bundle analysis.zip` on a benchmark
(R: `bundle="analysis.zip"`), or for an unchanged saved run:

```sh
omicsfold bundle --run previous/run.json --features features.tsv --metadata metadata.csv --output analysis.zip
```

The new ZIP contains exact input bytes, engine source and license, all original
report files, transitive runtime pins, checksums and restoration instructions.
Altered artifacts, environments or implementation are rejected. Bundles contain
the original data and require its data access controls. ZIP files are ignored by
Git. Source restoration still must pass the replay gate on compatible hardware.

```sh
omicsfold benchmark --features counts.tsv --metadata samples.csv --orientation samples-by-features --sample-id-column sample_id --outcome-column outcome --positive-label case --input-type counts --normalization median_ratio_log1p --missing-counts median --group-column patient --batch-column batch --weighting group_balanced --bootstrap-repeats 200 --output new-analysis
```

Supported software controls do not establish universal assay suitability,
statistical independence, or automatic cross-platform numerical equality.
