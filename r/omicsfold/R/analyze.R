#' Run the shared Python engine
#'
#' @param features Data frame with explicit character sample_id and numeric features.
#' @param metadata Data frame with character sample_id, outcome and optional groups.
#' @param positive_label Explicit positive outcome string.
#' @param output New output directory, which must not exist.
#' @param group_columns Character vector of metadata grouping column names.
#' @param folds Number of cross-validation folds.
#' @param seed Reproducible integer seed.
#' @param top_k Optional number of features selected within each training fold.
#' @param python Python executable containing the installed omicsfold package.
#' @return List containing metrics, provenance, folds, predictions and exploratory rankings.
#' @export
omicsfold_analyze <- function(features, metadata, positive_label, output,
                             group_columns = character(), folds = 5L,
                             seed = 42L, top_k = NULL,
                             python = Sys.getenv("OMICSFOLD_PYTHON", "python")) {
  if (!is.data.frame(features) || !is.data.frame(metadata))
    stop("features and metadata must be data frames")
  if (!is.character(features$sample_id) || !is.character(metadata$sample_id) ||
      !is.character(metadata$outcome))
    stop("sample_id and outcome must be explicit character columns")
  if (anyNA(metadata)) stop("metadata must not contain missing values")
  if (length(positive_label) != 1L || is.na(positive_label) || !nzchar(positive_label))
    stop("positive_label must be one nonempty string")
  if (length(output) != 1L || is.na(output) || !nzchar(output))
    stop("output must be one nonempty path")
  temporary <- tempfile("omicsfold-")
  dir.create(temporary)
  on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
  feature_path <- file.path(temporary, "features.csv")
  metadata_path <- file.path(temporary, "metadata.csv")
  utils::write.csv(features, feature_path, row.names = FALSE, na = "")
  utils::write.csv(metadata, metadata_path, row.names = FALSE, na = "")
  args <- c("-m", "omicsfold", "--features", feature_path, "--metadata", metadata_path,
            "--orientation", "samples-by-features", "--sample-id-column", "sample_id",
            "--outcome-column", "outcome", "--input-type", "numeric",
            "--positive-label", positive_label, "--output", output,
            "--folds", as.character(folds), "--seed", as.character(seed))
  if (length(group_columns) >= 1L) args <- c(args, "--group-column", group_columns[1L])
  if (length(group_columns) > 1L) {
    for (column in group_columns[-1L]) args <- c(args, "--additional-group-column", column)
  }
  if (!is.null(top_k)) args <- c(args, "--top-k", as.character(top_k))
  run <- .omicsfold_run(python, args)
  if (run$status != 0L)
    stop(paste0("OmicsFold Python engine failed:\n", run$stdout, run$stderr), call. = FALSE)
  result <- lapply(c("metrics", "provenance", "folds"), function(name)
    jsonlite::fromJSON(file.path(output, paste0(name, ".json"))))
  names(result) <- c("metrics", "provenance", "folds")
  result$predictions <- utils::read.csv(file.path(output, "predictions.csv"),
                                      colClasses = c(sample_id = "character"))
  result$exploratory_rankings <- utils::read.csv(file.path(output, "exploratory_rankings.csv"))
  result
}
