# Run an executable directly: processx supplies an argument vector without a shell.
.omicsfold_run <- function(executable, args) {
  tryCatch(processx::run(executable, args, error_on_status = FALSE,
                         windows_hide_window = TRUE, encoding = "UTF-8"),
           error = function(e) stop(paste0(
             "Cannot start Python OmicsFold using executable '", executable,
             "'. Select a Python environment with omicsfold installed (python -m pip install omicsfold), ",
             "or an installed omicsfold console script. Details: ", conditionMessage(e)), call. = FALSE))
}

.omicsfold_text <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(trimws(x)))
    stop(paste0(name, " must be one nonempty character string"), call. = FALSE)
  x
}

.omicsfold_numeric <- function(frame) {
  for (column in seq_along(frame)) {
    value <- frame[[column]]
    if (is.logical(value) && all(is.na(value))) value <- as.numeric(value)
    if (!is.numeric(value) || is.complex(value))
      stop("Feature values must be real numeric values or missing", call. = FALSE)
    # Explicit 17-digit serialization avoids losing double precision on the R
    # to Python boundary. Missing values remain empty cells, never literal IDs.
    frame[[column]] <- ifelse(is.na(value), NA_character_, sprintf("%.17g", value))
  }
  frame
}

.omicsfold_tsv <- function(frame, path) {
  utils::write.table(frame, path, sep = "\t", row.names = FALSE, col.names = TRUE,
                     quote = TRUE, qmethod = "double", na = "", eol = "\n",
                     fileEncoding = "UTF-8")
}

.omicsfold_read_tsv <- function(path, character_columns) {
  frame <- utils::read.delim(path, colClasses = "character", check.names = FALSE,
                            stringsAsFactors = FALSE, na.strings = NULL,
                            quote = "\"", comment.char = "", fileEncoding = "UTF-8")
  for (column in setdiff(names(frame), character_columns)) {
    value <- frame[[column]]
    value[value == ""] <- NA_character_
    frame[[column]] <- as.numeric(value)
  }
  frame
}

#' Benchmark an R Matrix or Data Frame Through the Installed Python Engine
#' @export
omicsfold_benchmark <- function(features, metadata, positive_label, input_type,
                               output, orientation, sample_id_column, outcome_column,
                               group_column = NULL, cohort_column = NULL,
                               split = NULL, holdout_cohort = NULL, folds = 5L,
                               seed = 42L, dataset_label = "Dataset",
                               executable = Sys.getenv("OMICSFOLD_PYTHON", "python"),
                               executable_type = c("python", "cli"),
                               feature_id_column = "feature_id", normalization = "auto",
                               missing_counts = "error", batch_column = NULL,
                               weighting = "auto", bootstrap_repeats = NULL,
                               metadata_map = NULL, environment_lock = NULL, bundle = NULL,
                               models = "logistic_l2", tune = FALSE, inner_folds = 3L,
                               search_space = NULL, time_column = NULL, time_cutoff = NULL,
                               temporal_gap_days = 0) {
  executable_type <- match.arg(executable_type)
  orientation <- match.arg(orientation, c("samples-by-features", "features-by-samples"))
  input_type <- match.arg(input_type, c("counts", "expression", "numeric"))
  normalization <- match.arg(normalization, c("auto", "none", "cpm_log1p", "median_ratio_log1p", "log1p", "clr"))
  missing_counts <- match.arg(missing_counts, c("error", "median"))
  weighting <- match.arg(weighting, c("auto", "sample", "group_balanced"))
  if (!is.null(split)) split <- match.arg(split, c("stratified", "grouped", "cohort", "temporal"))
  if (!is.character(models) || !length(models) || anyNA(models) || anyDuplicated(models) ||
      !all(models %in% c("logistic_l2", "random_forest", "gaussian_nb")))
    stop("models must name supported model families", call. = FALSE)
  if (!is.logical(tune) || length(tune) != 1L || is.na(tune))
    stop("tune must be TRUE or FALSE", call. = FALSE)
  for (name in c("positive_label", "sample_id_column", "outcome_column", "output",
                 "dataset_label", "executable", "feature_id_column"))
    .omicsfold_text(get(name), name)
  for (name in c("group_column", "cohort_column", "holdout_cohort", "batch_column", "metadata_map", "environment_lock", "bundle",
                 "search_space", "time_column", "time_cutoff"))
    if (!is.null(get(name))) .omicsfold_text(get(name), name)
  link <- Sys.readlink(output)
  if (file.exists(output) || dir.exists(output) || (!is.na(link) && nzchar(link)))
    stop(paste0("Output destination already exists: ", output), call. = FALSE)
  if (!is.data.frame(metadata)) stop("metadata must be a data frame", call. = FALSE)
  if (anyDuplicated(names(metadata))) stop("Duplicate metadata column names", call. = FALSE)
  mapped_names <- names(metadata)
  if (!is.null(metadata_map)) {
    mapping <- jsonlite::fromJSON(metadata_map, simplifyVector = FALSE)$columns
    mapped_names <- vapply(names(metadata), function(name)
      if (name %in% names(mapping)) mapping[[name]] else name, character(1))
  }
  if (!sample_id_column %in% mapped_names || !outcome_column %in% mapped_names)
    stop("Metadata requires the declared sample ID and outcome columns", call. = FALSE)
  source_id <- names(metadata)[mapped_names == sample_id_column]
  if (length(source_id) != 1L) stop("Metadata ID mapping must be unique", call. = FALSE)
  if (is.factor(metadata[[source_id]]))
    metadata[[source_id]] <- as.character(metadata[[source_id]])
  if (!is.character(metadata[[source_id]]))
    stop("Metadata sample IDs must be character strings to preserve leading zeros", call. = FALSE)
  metadata[] <- lapply(metadata, function(value) if (inherits(value, "POSIXt"))
    format(value, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS6Z") else as.character(value))
  if (!is.matrix(features) && !is.data.frame(features))
    stop("features must be a matrix or data frame", call. = FALSE)
  if (is.null(colnames(features)) || anyNA(colnames(features)) ||
      any(!nzchar(colnames(features))) || anyDuplicated(colnames(features)))
    stop("Feature/sample column names must be explicit, nonempty and unique", call. = FALSE)
  row_ids <- rownames(features)
  implicit_rows <- is.data.frame(features) && .row_names_info(features, 1L) < 0L
  frame <- as.data.frame(features, optional = TRUE, stringsAsFactors = FALSE)
  if (orientation == "samples-by-features") {
    if (sample_id_column %in% names(frame)) {
      ids <- frame[[sample_id_column]]
      if (is.factor(ids)) ids <- as.character(ids)
      frame <- frame[, names(frame) != sample_id_column, drop = FALSE]
    } else {
      if (is.null(row_ids) || implicit_rows)
        stop("Supply explicit sample row names or a sample ID column", call. = FALSE)
      ids <- row_ids
    }
    if (!is.character(ids)) stop("Feature sample IDs must be character strings", call. = FALSE)
    frame <- .omicsfold_numeric(frame)
    frame <- cbind(stats::setNames(data.frame(ids, stringsAsFactors = FALSE), sample_id_column), frame)
  } else {
    if (feature_id_column %in% names(frame)) {
      ids <- as.character(frame[[feature_id_column]])
      frame <- frame[, names(frame) != feature_id_column, drop = FALSE]
    } else {
      if (is.null(row_ids) || implicit_rows)
        stop("Supply explicit feature row names or a feature ID column", call. = FALSE)
      ids <- row_ids
    }
    frame <- .omicsfold_numeric(frame)
    frame <- cbind(stats::setNames(data.frame(ids, stringsAsFactors = FALSE), feature_id_column), frame)
  }
  prefix <- if (executable_type == "python") c("-m", "omicsfold") else character()
  probe <- .omicsfold_run(executable, c(prefix, "benchmark", "--help"))
  diagnostic <- paste(probe$stdout, probe$stderr, sep = "\n")
  if (probe$status != 0L || !grepl("--positive-label", diagnostic, fixed = TRUE))
    stop(paste0("Python OmicsFold is missing or does not support the benchmark CLI. ",
                "Install omicsfold in the selected Python environment ",
                "(python -m pip install omicsfold). Selected executable: ", executable,
                "\n", diagnostic), call. = FALSE)
  temporary <- tempfile("omicsfold benchmark inputs ")
  dir.create(temporary)
  on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
  feature_path <- file.path(temporary, "feature matrix.tsv")
  metadata_path <- file.path(temporary, "sample metadata.tsv")
  .omicsfold_tsv(frame, feature_path)
  .omicsfold_tsv(metadata, metadata_path)
  args <- c(prefix, "benchmark", "--features", feature_path, "--metadata", metadata_path,
            "--orientation", orientation, "--sample-id-column", sample_id_column,
            "--outcome-column", outcome_column, "--positive-label", positive_label,
            "--input-type", input_type, "--output", output,
            "--feature-id-column", feature_id_column, "--folds", as.character(folds),
            "--seed", as.character(seed), "--dataset-label", dataset_label,
            "--normalization", normalization, "--missing-counts", missing_counts,
            "--weighting", weighting, "--models", models, "--inner-folds", as.character(inner_folds),
            "--temporal-gap-days", as.character(temporal_gap_days))
  if (tune) args <- c(args, "--tune")
  optional <- list("--group-column" = group_column, "--cohort-column" = cohort_column,
                   "--split" = split, "--holdout-cohort" = holdout_cohort,
                   "--batch-column" = batch_column, "--metadata-map" = metadata_map,
                   "--environment-lock" = environment_lock,
                   "--bundle" = bundle,
                   "--search-space" = search_space, "--time-column" = time_column, "--time-cutoff" = time_cutoff,
                   "--bootstrap-repeats" = if (!is.null(bootstrap_repeats)) as.character(bootstrap_repeats) else NULL)
  for (flag in names(optional)) if (!is.null(optional[[flag]])) args <- c(args, flag, optional[[flag]])
  run <- .omicsfold_run(executable, args)
  if (run$status != 0L)
    stop(paste0("OmicsFold benchmark failed (exit ", run$status, "):\n",
                run$stdout, run$stderr), call. = FALSE)
  output <- normalizePath(output, winslash = "/", mustWork = TRUE)
  provenance <- jsonlite::fromJSON(file.path(output, "run.json"), simplifyVector = FALSE)
  if (!identical(provenance$report_schema_version, "1.0"))
    stop("Unsupported Python OmicsFold report schema; expected 1.0", call. = FALSE)
  list(metrics = .omicsfold_read_tsv(file.path(output, "metrics.tsv"), c("scope", "model")),
       predictions = .omicsfold_read_tsv(file.path(output, "predictions.tsv"),
                                        c("sample_id", "outcome", "positive_label", "model")),
       splits = .omicsfold_read_tsv(file.path(output, "splits.tsv"),
                                   c("partition", "sample_id", "outcome", "group", "cohort")),
       report = file.path(output, "report.html"), output = output, run = provenance)
}
