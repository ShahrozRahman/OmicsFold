# Synthetic demonstration only. First run: python examples/synthetic.py
library(omicsfold)
features <- read.csv("examples/results/synthetic_features.csv", colClasses = c(sample_id = "character"))
metadata <- read.csv("examples/results/synthetic_metadata.csv", colClasses = "character")
result <- omicsfold_analyze(features, metadata, positive_label = "case",
                            output = tempfile("omicsfold-result-"),
                            group_columns = "patient", top_k = 5L)
print(result$metrics)
python_predictions <- read.csv("examples/results/predictions.csv", colClasses = c(sample_id = "character"))
stopifnot(isTRUE(all.equal(result$predictions, python_predictions, tolerance = 1e-12)))
