# Same synthetic count fixture as the Python benchmark README command.
# Run after installing the development R archive and Python package.
library(omicsfold)
x <- read.delim(system.file("extdata", "counts.tsv", package = "omicsfold"),
                 colClasses = c(sample_id = "character"), check.names = FALSE)
metadata <- read.csv(system.file("extdata", "metadata.csv", package = "omicsfold"),
                      colClasses = "character", check.names = FALSE)
rownames(x) <- x$sample_id
matrix <- as.matrix(x[, names(x) != "sample_id", drop = FALSE])
arguments <- commandArgs(trailingOnly = TRUE)
output <- if (length(arguments)) arguments[1L] else file.path("examples", "results", "r-benchmark")
result <- omicsfold_benchmark(matrix, metadata, positive_label = "case", input_type = "counts",
                              output = output,
                              orientation = "samples-by-features", sample_id_column = "sample_id",
                              outcome_column = "outcome", group_column = "patient", cohort_column = "cohort",
                              split = "cohort", holdout_cohort = "external", seed = 42L,
                              dataset_label = "Synthetic count fixture (8 samples)")
print(result$metrics)
cat("HTML report:", result$report, "\n")
