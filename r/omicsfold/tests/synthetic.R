# All samples, features and outcomes in this test are synthetic.
library(omicsfold)

python <- Sys.getenv("OMICSFOLD_PYTHON", "python")
probe <- tryCatch(processx::run(python, c("-m", "omicsfold", "benchmark", "--help"),
  error_on_status = FALSE, windows_hide_window = TRUE), error = function(e) NULL)
available <- !is.null(probe) && probe$status == 0L &&
  grepl("--positive-label", probe$stdout, fixed = TRUE)
if (!available && nzchar(Sys.getenv("OMICSFOLD_REQUIRE_PYTHON")))
  stop("Python OmicsFold is required for this integration test run")

run_tests <- function() {
n <- 40L
features <- data.frame(sample_id = sprintf("%03d", seq_len(n)),
                       x = sin(seq_len(n)), z = cos(seq_len(n)))
metadata <- data.frame(sample_id = features$sample_id,
                       outcome = rep(c("control", "case"), n / 2L),
                       patient = rep(sprintf("P%02d", seq_len(n / 2L)), each = 2L))
output <- tempfile("omicsfold-r-test-")
result <- omicsfold_analyze(features, metadata, "case", output,
                            group_columns = "patient", folds = 4L, python = python)
stopifnot(nrow(result$predictions) == n,
          identical(result$predictions$sample_id, features$sample_id),
          result$provenance$positive_label == "case",
          length(result$metrics$roc_auc) == 1L)
assignments <- merge(result$predictions, metadata, by = "sample_id")
stopifnot(all(vapply(split(assignments$fold, assignments$patient),
                     function(x) length(unique(x)) == 1L, logical(1))))
repeat_output <- tempfile("omicsfold-r-repeat-")
repeated <- omicsfold_analyze(features[n:1L, ], metadata[n:1L, ], "case", repeat_output,
                              group_columns = "patient", folds = 4L, python = python)
stopifnot(identical(result$predictions, repeated$predictions),
          identical(result$metrics, repeated$metrics))
stopifnot(inherits(try(omicsfold_analyze(features, metadata, "case", output),
                       silent = TRUE), "try-error"))
unlink(c(output, repeat_output), recursive = TRUE)

}
if (available) run_tests() else message("Skipping integration tests: Python OmicsFold is unavailable")
