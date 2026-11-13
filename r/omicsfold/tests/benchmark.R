# Integration tests use only synthetic fixtures and installed Python OmicsFold.
library(omicsfold)

python <- Sys.getenv("OMICSFOLD_PYTHON", "python")
probe <- tryCatch(processx::run(python, c("-m", "omicsfold", "benchmark", "--help"),
  error_on_status = FALSE, windows_hide_window = TRUE), error = function(e) NULL)
available <- !is.null(probe) && probe$status == 0L &&
  grepl("--positive-label", probe$stdout, fixed = TRUE)
if (!available && nzchar(Sys.getenv("OMICSFOLD_REQUIRE_PYTHON")))
  stop("Python OmicsFold is required for this integration test run")

run_tests <- function() {
  temporary <- tempfile("omicsfold R checks with spaces ")
  dir.create(temporary)
  on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
  python <- Sys.getenv("OMICSFOLD_PYTHON", "python")
  feature_file <- system.file("extdata", "counts.tsv", package = "omicsfold")
  metadata_file <- system.file("extdata", "metadata.csv", package = "omicsfold")
  x <- read.delim(feature_file, colClasses = c(sample_id = "character"), check.names = FALSE)
  metadata <- read.csv(metadata_file, colClasses = "character", check.names = FALSE)
  rownames(x) <- x$sample_id
  matrix <- as.matrix(x[, names(x) != "sample_id", drop = FALSE])
  options <- list(metadata = metadata, positive_label = "case", input_type = "counts",
                  orientation = "samples-by-features", sample_id_column = "sample_id",
                  outcome_column = "outcome", group_column = "patient", cohort_column = "cohort",
                  split = "cohort", holdout_cohort = "external", seed = 42L,
                  dataset_label = "Synthetic count fixture (8 samples)", executable = python)
  r_output <- file.path(temporary, "R output with spaces")
  actual <- do.call(omicsfold_benchmark, c(list(features = matrix, output = r_output), options))
  stopifnot(is.data.frame(actual$metrics), is.data.frame(actual$predictions),
            is.data.frame(actual$splits), file.exists(actual$report),
            identical(actual$run$report_schema_version, "1.0"),
            identical(sort(unique(actual$splits$sample_id)), sprintf("%03d", 1:8)))
  # Independent direct installed Python CLI invocation on the original fixture.
  python_output <- file.path(temporary, "Python output with spaces")
  args <- c("-m", "omicsfold", "benchmark", "--features", feature_file, "--metadata", metadata_file,
            "--orientation", "samples-by-features", "--sample-id-column", "sample_id",
            "--outcome-column", "outcome", "--positive-label", "case", "--input-type", "counts",
            "--group-column", "patient", "--cohort-column", "cohort", "--split", "cohort",
            "--holdout-cohort", "external", "--seed", "42",
            "--dataset-label", options$dataset_label, "--output", python_output)
  direct <- processx::run(python, args, error_on_status = FALSE, windows_hide_window = TRUE)
  stopifnot(direct$status == 0L)
  read_result <- function(name) read.delim(file.path(python_output, name),
    colClasses = "character", check.names = FALSE, na.strings = NULL)
  expected_splits <- read_result("splits.tsv")
  expected_predictions <- read_result("predictions.tsv")
  stopifnot(identical(actual$splits$sample_id, expected_splits$sample_id),
            identical(actual$splits$partition, expected_splits$partition),
            identical(actual$splits$fold, as.numeric(expected_splits$fold)),
            identical(actual$predictions$sample_id, expected_predictions$sample_id),
            isTRUE(all.equal(actual$predictions$positive_probability,
              as.numeric(expected_predictions$positive_probability), tolerance = 1e-12)))
  expected_metrics <- read_result("metrics.tsv")
  for (column in c("average_precision", "roc_auc", "test_positive_prevalence", "test_sample_count"))
    stopifnot(isTRUE(all.equal(actual$metrics[[column]], as.numeric(expected_metrics[[column]]), tolerance = 1e-12)))
  stopifnot(identical(sort(list.files(actual$output)), sort(list.files(python_output))))
  python_run <- jsonlite::fromJSON(file.path(python_output, "run.json"), simplifyVector = FALSE)
  stopifnot(identical(actual$run$configuration, python_run$configuration),
            identical(actual$run$input_checksums$canonical_features_sha256,
                      python_run$input_checksums$canonical_features_sha256),
            identical(actual$run$input_checksums$canonical_metadata_sha256,
                      python_run$input_checksums$canonical_metadata_sha256))
  r_coefficients <- read.delim(file.path(actual$output, "features.tsv"))
  python_coefficients <- read.delim(file.path(python_output, "features.tsv"))
  stopifnot(isTRUE(all.equal(r_coefficients, python_coefficients, tolerance = 1e-12)))
  # Data-frame and transposed-matrix paths must preserve the same named data.
  frame_result <- do.call(omicsfold_benchmark, c(list(features = x, output = file.path(temporary, "frame result")), options))
  transposed_options <- options
  transposed_options$orientation <- "features-by-samples"
  transpose_result <- do.call(omicsfold_benchmark, c(list(features = t(matrix), output = file.path(temporary, "transpose result")), transposed_options))
  stopifnot(identical(actual$splits, frame_result$splits), identical(actual$splits, transpose_result$splits),
            isTRUE(all.equal(actual$predictions, frame_result$predictions, tolerance = 1e-12)),
            isTRUE(all.equal(actual$predictions, transpose_result$predictions, tolerance = 1e-12)))
  # Shell metacharacters, quotes and nonsyntactic names must remain literal.
  weird <- matrix
  colnames(weird) <- c("gene with spaces", "gene; echo SHOULD_NOT_RUN", "gene \"quoted\"")
  weird_options <- options
  rownames(weird)[1:2] <- c("NA", "sample 002 & quoted \"x\"")
  weird_options$metadata$sample_id[weird_options$metadata$sample_id == "001"] <- "NA"
  weird_options$metadata$sample_id[weird_options$metadata$sample_id == "002"] <- "sample 002 & quoted \"x\""
  weird_options$dataset_label <- "Dataset \"quoted\" & echo SHOULD_NOT_RUN; $(touch SHOULD_NOT_RUN)"
  named <- do.call(omicsfold_benchmark, c(list(features = weird, output = file.path(temporary, "literal name result")), weird_options))
  names_table <- read.delim(file.path(named$output, "features.tsv"), check.names = FALSE, quote = "\"")
  stopifnot(identical(names_table$feature, colnames(weird)),
            identical(named$run$dataset_label, weird_options$dataset_label),
            identical(sort(unique(named$splits$sample_id)), sort(rownames(weird))),
            !file.exists("SHOULD_NOT_RUN"))
  # Test the console executable route as well as Python -m.
  find_cli <- processx::run(python, c("-c", paste0(
    "import pathlib,sys,sysconfig; dirs=[pathlib.Path(sysconfig.get_path('scripts')),pathlib.Path(sys.base_prefix)/'Scripts',pathlib.Path(sys.base_prefix)/'bin']; ",
    "print(next((str(p/n) for p in dirs for n in ('omicsfold.exe','omicsfold') if (p/n).is_file()),''))")),
    windows_hide_window = TRUE)
  console <- trimws(find_cli$stdout)
  if (nzchar(console)) {
    console_options <- options
    console_options$executable <- console
    console_options$executable_type <- "cli"
    console_result <- do.call(omicsfold_benchmark, c(list(features = matrix, output = file.path(temporary, "console result")), console_options))
    stopifnot(isTRUE(all.equal(actual$predictions, console_result$predictions, tolerance = 1e-12)))
  } else {
    stop("Installed Python package did not provide the omicsfold console script")
  }
  # Missing executable and missing Python package produce actionable diagnostics.
  missing_options <- options
  missing_options$executable <- file.path(temporary, "no such executable")
  missing <- tryCatch(do.call(omicsfold_benchmark, c(list(features = matrix, output = file.path(temporary, "missing executable")), missing_options)), error = identity)
  stopifnot(inherits(missing, "error"), grepl("Cannot start Python OmicsFold", conditionMessage(missing), fixed = TRUE))
  environment <- file.path(temporary, "Python without OmicsFold")
  create <- processx::run(python, c("-m", "venv", "--without-pip", environment), windows_hide_window = TRUE)
  stopifnot(create$status == 0L)
  missing_options$executable <- if (.Platform$OS.type == "windows") file.path(environment, "Scripts", "python.exe") else file.path(environment, "bin", "python")
  missing <- tryCatch(do.call(omicsfold_benchmark, c(list(features = matrix, output = file.path(temporary, "missing package")), missing_options)), error = identity)
  stopifnot(inherits(missing, "error"), grepl("Python OmicsFold is missing", conditionMessage(missing), fixed = TRUE),
            grepl("No module named omicsfold", conditionMessage(missing), fixed = TRUE))
  existing <- tryCatch(do.call(omicsfold_benchmark, c(list(features = matrix, output = r_output), options)), error = identity)
  stopifnot(inherits(existing, "error"), grepl("already exists", conditionMessage(existing), fixed = TRUE))
  cat("R/Python benchmark parity passed: schema 1.0, exact splits, scores/coefficients within 1e-12.\n")
}

if (available) run_tests() else message("Skipping integration tests: Python OmicsFold is unavailable")
