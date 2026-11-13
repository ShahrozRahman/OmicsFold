library(omicsfold)

expect_error <- function(expression, pattern) {
  error <- tryCatch(force(expression), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), fixed = TRUE))
}

# These checks run without Python and reject invalid inputs before launching it.
metadata <- data.frame(sample_id = c("001", "002"), outcome = c("case", "control"))
features <- data.frame(sample_id = metadata$sample_id, value = c(1, 2))
options <- list(features = features, metadata = metadata, positive_label = "case",
                input_type = "numeric", output = tempfile("validation-"),
                orientation = "samples-by-features", sample_id_column = "sample_id",
                outcome_column = "outcome", executable = "no-python-needed")
invalid <- options
invalid$positive_label <- ""
expect_error(do.call(omicsfold_benchmark, invalid), "positive_label must be one nonempty")
invalid <- options
invalid$metadata$sample_id <- 1:2
expect_error(do.call(omicsfold_benchmark, invalid), "Metadata sample IDs must be character")
invalid <- options
invalid$features$value <- c("one", "two")
expect_error(do.call(omicsfold_benchmark, invalid), "Feature values must be real numeric")
invalid <- options
invalid$features$sample_id <- 1:2
expect_error(do.call(omicsfold_benchmark, invalid), "Feature sample IDs must be character")
expect_error(omicsfold_analyze(NULL, metadata, "case", tempfile()),
             "features and metadata must be data frames")
