# Synthetic nested/temporal parity through the installed Python engine.
library(omicsfold)
python <- Sys.getenv("OMICSFOLD_PYTHON", "python")
probe <- tryCatch(processx::run(python, c("-m", "omicsfold", "benchmark", "--help"),
  error_on_status = FALSE, windows_hide_window = TRUE), error = function(e) NULL)
available <- !is.null(probe) && probe$status == 0L && grepl("--tune", probe$stdout, fixed = TRUE)
if (!available && nzchar(Sys.getenv("OMICSFOLD_REQUIRE_PYTHON")))
  stop("Python OmicsFold with nested selection is required")

run_tests <- function() {
  temporary <- tempfile("nested temporal parity ")
  dir.create(temporary)
  on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
  n <- 72L
  i <- seq_len(n)
  x <- data.frame(sample_id = sprintf("S%03d", i), synthetic_x = i %% 7L,
                   synthetic_y = (i * 3L) %% 11L + (i %% 2L) * 2L, check.names = FALSE)
  m <- data.frame(sample_id = x$sample_id, outcome = rep(c("case", "control"), n / 2L),
                   patient = sprintf("P%03d", (i - 1L) %/% 2L),
                   collected_at = as.Date("2020-01-01") + (i - 1L) %/% 2L)
  r_output <- file.path(temporary, "R results")
  actual <- omicsfold_benchmark(x, m, "case", "numeric", r_output,
    orientation = "samples-by-features", sample_id_column = "sample_id", outcome_column = "outcome",
    group_column = "patient", split = "temporal", time_column = "collected_at",
    time_cutoff = "2020-01-29", temporal_gap_days = 1, tune = TRUE,
    models = c("logistic_l2", "gaussian_nb"), inner_folds = 3L,
    bootstrap_repeats = 40L, executable = python)
  feature_path <- file.path(temporary, "features.tsv")
  metadata_path <- file.path(temporary, "metadata.tsv")
  utils::write.table(x, feature_path, sep = "\t", row.names = FALSE, quote = TRUE)
  utils::write.table(m, metadata_path, sep = "\t", row.names = FALSE, quote = TRUE)
  direct_output <- file.path(temporary, "Python results")
  args <- c("-m", "omicsfold", "benchmark", "--features", feature_path, "--metadata", metadata_path,
    "--orientation", "samples-by-features", "--sample-id-column", "sample_id", "--outcome-column", "outcome",
    "--positive-label", "case", "--input-type", "numeric", "--group-column", "patient",
    "--split", "temporal", "--time-column", "collected_at", "--time-cutoff", "2020-01-29",
    "--temporal-gap-days", "1", "--tune", "--models", "logistic_l2", "gaussian_nb",
    "--inner-folds", "3", "--bootstrap-repeats", "40", "--output", direct_output)
  processx::run(python, args, windows_hide_window = TRUE)
  expected <- jsonlite::fromJSON(file.path(direct_output, "run.json"), simplifyVector = FALSE)
  stopifnot(identical(actual$run$result_checksums, expected$result_checksums),
            isTRUE(actual$run$validation_design$nested_selection),
            actual$run$folds[[1L]]$temporal$latest_train < actual$run$folds[[1L]]$temporal$earliest_test)
  cat("Nested temporal R/Python exact checksum parity passed\n")
}

if (available) run_tests() else cat("Skipping nested integration: Python OmicsFold is unavailable\n")
