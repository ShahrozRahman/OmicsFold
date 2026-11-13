\name{omicsfold_analyze}
\alias{omicsfold_analyze}
\title{Run the Shared OmicsFold Python Engine}
\usage{
omicsfold_analyze(features, metadata, positive_label, output,
  group_columns = character(), folds = 5L, seed = 42L,
  top_k = NULL, python = Sys.getenv("OMICSFOLD_PYTHON", "python"))
}
\arguments{
\item{features}{Data frame with character sample_id and numeric feature columns.}
\item{metadata}{Data frame with character sample_id, outcome, and optional groups.}
\item{positive_label}{Explicit positive outcome string.}
\item{output}{New output directory. Existing directories are rejected.}
\item{group_columns}{Metadata columns whose shared values keep samples together.}
\item{folds}{Number of folds, at least two.}
\item{seed}{Integer random seed.}
\item{top_k}{Optional number of features selected inside each training fold.}
\item{python}{Python executable with omicsfold installed.}
}
\description{Runs the installed Python engine using temporary CSV inputs.
No preprocessing or statistical analysis is implemented independently in R.}
\value{List with metrics, provenance, folds, predictions and exploratory rankings.}
