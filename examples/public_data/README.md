# Public-data case study: historical ClinVar disagreement

This case study evaluates recorded classification disagreement in a historical
ClinVar Conflict Explorer snapshot using numeric variant features and gene-grouped
validation. It illustrates a reproducible research workflow, not pathogenicity
prediction. The expression-cohort compatibility assessment below documents why
those expression datasets are not used for a cross-cohort performance claim.

Open the [published self-contained report](published/report.html). Its source
and interpretation appendix is included in the HTML, which needs no server.
[Fold metrics](published/metrics.tsv), [exploratory coefficients](published/features.tsv),
[preparation provenance](published/preparation.json), and the independently checked
[split audit](published/split_audit.json) accompany it. This is real public data,
not a synthetic biological example.

## Reproduce with the installed CLI

From the repository root, install the package, then run:

```sh
python -m pip install .
python examples/public_data/run_case_study.py --download
```

The script resolves the **installed `omicsfold` console entry point**, calls
`omicsfold validate` followed by `omicsfold benchmark`, and independently inspects
the resulting splits, genes, outcome counts and held-out predictions. It does
not call the Python benchmark API as a substitute for exercising the CLI.
The model/split settings are in [benchmark.json](benchmark.json).

Raw downloads, prepared matrices and complete benchmark results live outside
Git: `%LOCALAPPDATA%/OmicsFold/public-data/` on Windows, or
`~/.cache/OmicsFold/public-data/` elsewhere. Default run work is
`clinvar-case-study/` under that cache; it must not already exist. For another run:

```sh
python examples/public_data/run_case_study.py --download --work-dir /path/to/new/outside-repository-work
```

If Kaggle download access is unavailable, download/extract the source CSV
manually and supply `--raw-csv /path/to/clinvar_conflicting.csv`. The checksum is
always verified, regardless of download method. No credential is stored by this
script. Reusing an already downloaded CSV does not require network access.
Raw data are never written into this repository; inside-repository work/cache
locations are rejected. The compact derived publication defaults to
`WORK/publication`; an explicit unused `--publish-dir` can publish that compact
report inside the repository. Existing work/publication paths are rejected.

Preparation alone is reproducible with:

```sh
python examples/public_data/prepare.py --download --output /path/to/new/outside-repository-prepared
```

The full six-file benchmark schema is generated in `WORK/results/`. Large
variant-level `predictions.tsv`, `splits.tsv` and `run.json` stay outside Git.
The checked-in publication includes a compact `run_summary.json` and a checksum
manifest of the full CLI outputs instead. Its HTML adds a source/interpretation
appendix to the unmodified installed-CLI report; it does not change scores.
The manifest distinguishes original CLI artifacts from published artifacts.

## Source and outcome

Kevin Arvai, [Genetic Variant Classifications](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting),
Kaggle dataset `kevinarvai/clinvar-conflicting`, **version 2**, CC0: Public Domain.
The [dataset metadata API](https://www.kaggle.com/api/v1/datasets/view/kevinarvai/clinvar-conflicting)
identifies its upstream source as the ClinVar GRCh37 VCF downloaded on
**2018-04-07**; the Kaggle version was updated in 2020. This is a historical
derived table, not a current ClinVar release. Author preparation code is in
[arvkevi/clinvar-kaggle](https://github.com/arvkevi/clinvar-kaggle). Original
resource: [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/).

The exact CSV SHA-256 is:

```text
53061e207d437af397028760be5bc566d216f6932e21db5a10322b843c41689a
```

This matches the checksum recorded by the sibling ClinVar Conflict Explorer
project, whose README and training code were inspected. The CSV itself is not
vendored here. The dataset author restricts records to variants with multiple
submissions. `CLASS=1` maps to **recorded classification disagreement** and
`CLASS=0` to **recorded agreement**. The author's categories combine benign with
likely benign, VUS as its own category, and pathogenic with likely pathogenic.
This label is disagreement in that snapshot. It is **not pathogenicity**, a
patient phenotype, a prediction of future disagreement, or a clinical score.
Current [ClinVar classification documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig/)
describes aggregation/conflict semantics; the author-defined historical label
is preserved rather than recomputed using today's rules.

## Inclusion, feature mapping and preprocessing

- Start with 65,188 rows from the checksum-pinned snapshot.
- Normalize `SYMBOL` by trimming whitespace and uppercasing. Exclude 16 records
  with missing/invalid single gene symbols; require `[A-Z0-9_.-]+`. No ambiguous
  multi-gene mapping or external remapping is performed. This leaves 65,172
  variants across 2,328 gene groups.
- Require complete, unique `CHROM:POS:REF:ALT` keys and binary source labels.
  Keys identify variants in the source GRCh37 snapshot, not new ClinVar accessions.
- Explicit predictor allowlist: `AF_ESP`, `AF_EXAC`, `AF_TGP`. Each maps to
  `log1p(1,000,000 Ã— original frequency)`, independently per value. Require
  finite frequencies in [0,1] or missing values. There are no missing values in
  the three selected columns in this snapshot; handling remains specified.
- `SYMBOL` becomes metadata `gene` for splitting, never a predictor. Clinical
  assertions, review information, variant coordinates/IDs and categorical
  annotations are excluded from the matrix. `CLASS` supplies metadata only.
- The installed OmicsFold numeric pipeline fits median imputation, constant
  feature removal and standard scaling inside each training fold. No global
  fitted transform is used during preparation.
- Compare prevalence with C=1 L2 logistic regression, liblinear solver, tolerance
  1e-4, maximum 2,000 iterations, no class weights. Five stratified group folds,
  gene grouping, seed 42, positive label `recorded_disagreement`. There is no
  tuning, category encoding, missingness-indicator feature or temporal holdout.
  This reuses the prior project's data/question, not its entire multi-model
  training pipeline.

## Inspected splits and interpretation

This historical publication deliberately uses `weighting=sample` and disables
bootstrap intervals in its configuration to preserve its original estimand.
The current benchmark default instead uses equal group contribution and whole-group
intervals. To evaluate that alternative, change these explicit configuration
fields and produce a new report; do not treat the old numbers below as the new
group-balanced results. Published manifests refer to the scripts/configuration
at the historical run revision, not later edits.

There are 48,749 agreement and 16,423 disagreement variants (positive prevalence
0.251995). All three features survive preprocessing in every fold. The audit
confirms disjoint IDs/genes, both classes on both sides, all eligible samples
accounted for, and exactly one held-out prediction per variant/model.

| Fold | Training variants | Test variants | Test genes | Test positives |
| --- | ---: | ---: | ---: | ---: |
| 0 | 52,138 | 13,034 | 457 | 3,284 |
| 1 | 52,136 | 13,036 | 462 | 3,286 |
| 2 | 52,138 | 13,034 | 445 | 3,284 |
| 3 | 52,137 | 13,035 | 480 | 3,285 |
| 4 | 52,139 | 13,033 | 484 | 3,284 |

| Model | Mean average precision | Mean ROC AUC |
| --- | ---: | ---: |
| Training-prevalence baseline | 0.251995 | 0.500000 |
| Frequency-only L2 logistic regression | 0.299413 | 0.574151 |

The model improves mean AP by **0.047418** absolute and AUC by **0.074151**.
This is a modest internal ranking signal. It does not validate individual
variants or clinical decisions. Gene grouping limits direct gene reuse across
partitions but does not eliminate ascertainment, annotation or other correlated
sources of bias. Metrics are sample-level (here, variant-level), so genes with
many variants have more weight. Means are unweighted fold means. The report's
PR curve is **individual fold 0 only**, not a pooled curve. Coefficient rankings
remain exploratory; no confidence intervals or external/temporal validation
are provided.

## Expression-cohort gate: pending

The reproducible inspection and exact matrix checksums are recorded in
[expression_assessment.json](expression_assessment.json). To inspect current
downloaded releases again without running any expression benchmark:

```sh
python examples/public_data/assess_expression.py --output /path/to/new-expression-assessment.json
```

| Criterion | GSE10072 | GSE7670 |
| --- | --- | --- |
| Accession | [GSE10072](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10072) | [GSE7670](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7670) |
| Platform | GPL96, HG-U133A | GPL96, HG-U133A |
| Included tissue labels | 58 adenocarcinoma tumors, 49 non-tumor tissues | 26 adenocarcinoma tumors, 26 adjacent non-tumor tissues |
| Patient mapping | `GT...` suffix in sample title; 74 source patient identifiers | Number preceding `N`/`T` in sample title; 26 source patient identifiers |
| Inclusion/exclusion | All 107 released final tissue profiles; source already averaged technical replicates | Only numbered adenocarcinoma pairs; exclude large-cell pair, mixtures, commercial normals and cell lines (14 records) |
| Released processing | RMA, with technical replicate averaging described by series | MAS5, per-array target intensity 500 |

The common phenotype would be tissue tumor status, **not cancer versus healthy
people**. Both released tables contain 22,283 unique probes. A defensible candidate
feature mapping is exact same-platform GPL96 probe-set ID intersection excluding
`AFFX-` controls: **22,215 common probes**, without symbol collapse or supervised
feature selection. Sample metadata and these probe checks pass.

The processing gate does **not** pass: RMA values and MAS5 intensities have not
been put through a verified common workflow. Method provenance is directly
documented in the [GSE10072 sample record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM254700)
and [GSE7670 sample record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM185812).
We did not simply merge, jointly renormalize, or batch-correct the released
matrices and call that cross-cohort validation. This is a conservative readiness
decision, not evidence that the studies can never be compared. A future case
study must establish compatible raw-CEL or validated training-reference
processing, confirm cohort/patient independence and probe annotation, and fit
all multi-sample parameters using training samples only. The preprocessing gate
is an explicitly documented manual scientific review, not an automated proof
of cohort compatibility.

The assessment script downloads only the small released matrices to the external
cache. It does not download/reprocess raw CEL files. GEO study citations:
GSE10072, Landi et al., [PMID 18297132](https://pubmed.ncbi.nlm.nih.gov/18297132/);
GSE7670, Su et al., [PMID 17540040](https://pubmed.ncbi.nlm.nih.gov/17540040/).
No expression data have been redistributed in this repository.
