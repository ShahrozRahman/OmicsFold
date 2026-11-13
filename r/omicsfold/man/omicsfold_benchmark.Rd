\name{omicsfold_benchmark}
\alias{omicsfold_benchmark}
\title{Benchmark R Inputs Through the Installed Python CLI}
\usage{
omicsfold_benchmark(features, metadata, positive_label, input_type,
  output, orientation, sample_id_column, outcome_column,
  group_column = NULL, cohort_column = NULL, split = NULL,
  holdout_cohort = NULL, folds = 5L, seed = 42L,
  dataset_label = "Dataset",
  executable = Sys.getenv("OMICSFOLD_PYTHON", "python"),
  executable_type = c("python", "cli"), feature_id_column = "feature_id",
  normalization = "auto", missing_counts = "error", batch_column = NULL,
  weighting = "auto", bootstrap_repeats = NULL, metadata_map = NULL,
  environment_lock = NULL, bundle = NULL, models = "logistic_l2",
  tune = FALSE, inner_folds = 3L, search_space = NULL,
  time_column = NULL, time_cutoff = NULL, temporal_gap_days = 0)
}
\arguments{
\item{features}{Real numeric matrix or data frame. Explicit sample/feature names are required; missing numerical values are allowed.}
\item{metadata}{Separate metadata data frame, including explicit character sample IDs and outcomes.}
\item{positive_label}{User-specified positive outcome string.}
\item{input_type}{One of counts, expression, numeric.}
\item{output}{New result directory; existing paths are rejected.}
\item{orientation}{Explicitly samples-by-features or features-by-samples.}
\item{sample_id_column}{Metadata ID column; also an optional sample-row data-frame ID column.}
\item{outcome_column}{Metadata outcome column.}
\item{group_column}{Optional metadata grouping column.}
\item{cohort_column}{Optional metadata cohort column.}
\item{split}{Optional stratified, grouped, cohort or temporal split. NULL uses the CLI default.}
\item{holdout_cohort}{Optional named test cohort.}
\item{folds}{Cross-validation fold count; ignored for cohort holdout.}
\item{seed}{Integer random seed.}
\item{dataset_label}{Display label, escaped by the Python report renderer.}
\item{executable}{Path/name of Python or the installed omicsfold console executable, without arguments or shell quoting.}
\item{executable_type}{python invokes the executable with -m omicsfold; cli invokes the console script directly.}
\item{feature_id_column}{Optional feature-row data-frame ID column; otherwise use explicit feature row names.}
\item{normalization}{auto, none, cpm_log1p, median_ratio_log1p, log1p or clr; all fitted parameters use training samples only.}
\item{missing_counts}{error by default; median explicitly enables experimental training-median count imputation.}
\item{batch_column}{Optional metadata batch column for training-fitted, phenotype-protected additive batch adjustment. Unseen test batches are rejected.}
\item{weighting}{auto uses equal group contribution when groups are supplied; sample or group_balanced can be requested explicitly.}
\item{bootstrap_repeats}{Whole-group bootstrap repetitions, at least 40, or zero to disable. NULL uses 200 for grouped runs and zero otherwise.}
\item{metadata_map}{Optional JSON path with explicit metadata column and value harmonization maps. IDs are never relabeled.}
\item{environment_lock}{Optional prior run.json or environment JSON; incompatible environments are rejected.}
\item{bundle}{Optional new .zip path preserving original inputs, engine source, runtime pins and results.}
\item{models}{One or more of logistic_l2, random_forest, gaussian_nb. Multiple candidates require tuning.}
\item{tune}{Select candidates using inner validation confined to each outer training partition.}
\item{inner_folds}{Inner fold count for nested selection.}
\item{search_space}{Optional JSON path with a predeclared list of candidate model/parameter objects.}
\item{time_column}{Metadata date/time column for chronological validation; ISO timestamps or R Date/POSIXt values.}
\item{time_cutoff}{Optional ISO boundary for a single chronological holdout; otherwise forward CV is used.}
\item{temporal_gap_days}{Nonnegative embargo between training and the test boundary.}
}
\description{Serializes explicit names and 17-digit numerical values to temporary
UTF-8 TSV files and runs the installed Python benchmark engine. No model fitting
or preprocessing is reimplemented in R. Uses processx argument vectors without
a shell; paths with spaces and shell metacharacters are passed literally.}
\details{For sample-row matrices, row names are sample IDs and column names are
feature names. For feature-row matrices the roles are reversed. Data frames can
instead contain the declared ID column; otherwise automatic row names are rejected.
Metadata are matched by exact ID in Python. Character IDs such as 001 or NA are
preserved; numeric ID columns are rejected. Feature names are not syntactically
repaired. The output is benchmark report schema 1.0 with six files:
report.html, predictions.tsv, metrics.tsv, splits.tsv, features.tsv and run.json.
Metrics describe held-out samples with the selected weighting and specified positive label,
including unweighted fold means. Whole-group confidence intervals and fitted preprocessing
states are recorded in run.json. Retention/coefficient tables are exploratory outputs in features.tsv.
Python OmicsFold must be installed in the selected environment. R dependencies
are jsonlite and processx.}
\value{List containing metrics, predictions and splits data frames, report (the
absolute HTML path), output (the absolute result directory), and run (parsed provenance).}
\examples{
\dontrun{
x <- read.delim(system.file("extdata", "counts.tsv", package = "omicsfold"),
                colClasses = c(sample_id = "character"), check.names = FALSE)
m <- read.csv(system.file("extdata", "metadata.csv", package = "omicsfold"),
              colClasses = "character", check.names = FALSE)
result <- omicsfold_benchmark(x, m, "case", "counts", tempfile("benchmark "),
  orientation = "samples-by-features", sample_id_column = "sample_id",
  outcome_column = "outcome", group_column = "patient", cohort_column = "cohort",
  split = "cohort", holdout_cohort = "external")
result$metrics
}
}
