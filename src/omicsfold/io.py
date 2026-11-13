"""Strict CSV/TSV input validation without fitting transformations."""
import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


def _table(path, source_info=None):
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".tsv"):
        raise ValueError(f"Expected a .csv or .tsv file: {path}")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = path.read_bytes()
    if source_info is not None:
        source_info.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), delimiter=delimiter, strict=True))
    if not rows or len(rows) < 2:
        raise ValueError(f"Table must contain a header and data rows: {path}")
    header = rows[0]
    if any(not name.strip() for name in header):
        raise ValueError(f"Empty column name in {path}")
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate column/feature names in {path}: {duplicates}")
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError(f"Wrong number of fields in {path}, row {line}")
    return pd.DataFrame(rows[1:], columns=header)


def _unique(values, description):
    if values.map(lambda value: not value.strip()).any():
        raise ValueError(f"Missing {description}")
    duplicates = sorted(values[values.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(f"Duplicate {description}: {duplicates}")


@dataclass
class LoadedData:
    """Canonical sample-row features and separately aligned metadata."""
    features: pd.DataFrame
    metadata: pd.DataFrame
    positive_label: str
    input_type: str
    group_column: str | None
    cohort_column: str | None
    input_provenance: dict = field(default_factory=dict)
    alignment_audit: dict = field(default_factory=dict)
    loading_configuration: dict = field(default_factory=dict)

    def summary(self):
        return {"valid": True, "samples": len(self.features),
                "features": len(self.features.columns) - 1,
                "input_type": self.input_type, "positive_label": self.positive_label,
                "outcome_counts": self.metadata.outcome.value_counts().to_dict(),
                "missing_feature_values": int(self.features.drop(columns="sample_id").isna().sum().sum()),
                "group_column": self.group_column, "cohort_column": self.cohort_column}


def load_data(features_path, metadata_path, *, orientation, sample_id_column,
              outcome_column, positive_label, input_type, group_column=None,
              cohort_column=None, feature_id_column="feature_id", allow_missing_counts=False,
              metadata_mapping=None):
    """Load and validate, preserving exact IDs and matrix sample order.

    orientation is 'samples-by-features' or 'features-by-samples'. The latter
    uses feature_id_column for feature names; remaining headers are sample IDs.
    sample_id_column always names the metadata ID column. No rows are dropped.
    Numeric missing tokens are empty, NA, NaN, nan, and N/A; identifiers and
    outcomes are read literally (except blank outcomes are rejected).
    """
    if orientation not in ("samples-by-features", "features-by-samples"):
        raise ValueError("orientation must be samples-by-features or features-by-samples")
    if input_type not in ("counts", "expression", "numeric"):
        raise ValueError("input_type must be counts, expression, or numeric")
    roles = [sample_id_column, outcome_column] + [c for c in (group_column, cohort_column) if c is not None]
    if any(not isinstance(c, str) or not c.strip() for c in roles) or len(set(roles)) != len(roles):
        raise ValueError("ID, outcome, group and cohort columns must be distinct nonempty names")
    sources = {"features": {}, "metadata": {}}
    metadata = _table(metadata_path, sources["metadata"])
    source_metadata_columns = set(metadata.columns)
    harmonization = []
    if metadata_mapping is not None:
        if not isinstance(metadata_mapping, dict) or set(metadata_mapping) - {"columns", "values"}:
            raise ValueError("metadata_mapping accepts only explicit columns and values maps")
        columns = metadata_mapping.get("columns", {})
        if not isinstance(columns, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in columns.items()):
            raise ValueError("Metadata columns mapping must contain string names")
        if not isinstance(metadata_mapping.get("values", {}), dict):
            raise ValueError("Metadata values mapping must be an object")
        if any(c not in metadata for c in columns):
            raise ValueError("Metadata column mapping references missing columns")
        metadata = metadata.rename(columns=columns)
        if metadata.columns.duplicated().any():
            raise ValueError("Metadata column mapping creates duplicate names")
        harmonization.append({"column_mapping": columns})
        for column, mapping in metadata_mapping.get("values", {}).items():
            if column == sample_id_column:
                raise ValueError("Sample ID harmonization is prohibited; IDs must match exactly")
            if column not in metadata or not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
                raise ValueError("Invalid metadata value mapping")
            unmatched = sorted(set(metadata[column]) - set(mapping))
            if unmatched:
                raise ValueError(f"Unmapped metadata values in {column}: {unmatched}")
            metadata[column] = metadata[column].map(mapping)
            harmonization.append({"column": column, "value_mapping": mapping})
    for column in roles:
        if column not in metadata:
            raise ValueError(f"Metadata column not found: {column}")
    _unique(metadata[sample_id_column], "metadata sample IDs")
    if metadata[outcome_column].map(lambda x: not x.strip()).any():
        raise ValueError("Missing outcomes are not allowed")
    labels = set(metadata[outcome_column])
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError("Exactly two outcomes and a matching explicit positive label are required")
    for column in (group_column, cohort_column):
        if column is not None and metadata[column].map(lambda x: not x.strip()).any():
            raise ValueError(f"Missing values in metadata column: {column}")
    matrix = _table(features_path, sources["features"])
    if orientation == "samples-by-features":
        if sample_id_column not in matrix:
            raise ValueError(f"Matrix sample ID column not found: {sample_id_column}")
        ids = matrix[sample_id_column]
        values = matrix.drop(columns=sample_id_column)
    else:
        if feature_id_column not in matrix:
            raise ValueError(f"Matrix feature ID column not found: {feature_id_column}")
        _unique(matrix[feature_id_column], "feature names")
        values = matrix.set_index(feature_id_column).T.reset_index(drop=True)
        ids = pd.Series([c for c in matrix if c != feature_id_column])
    _unique(ids, "matrix sample IDs")
    if values.shape[1] == 0:
        raise ValueError("Matrix must contain at least one feature")
    forbidden = ((set(metadata.columns) | source_metadata_columns) - {sample_id_column}) | {"sample_id", "outcome"}
    overlap = sorted(set(values.columns) & forbidden)
    if overlap:
        raise ValueError(f"Metadata/reserved columns cannot be matrix features: {overlap}")
    matrix_only = sorted(set(ids) - set(metadata[sample_id_column]))
    metadata_only = sorted(set(metadata[sample_id_column]) - set(ids))
    if matrix_only or metadata_only:
        raise ValueError(f"Unmatched IDs: matrix-only={matrix_only}; metadata-only={metadata_only}")
    metadata_positions = {value: i for i, value in enumerate(metadata[sample_id_column])}
    alignment = {"matched_samples": len(ids), "matrix_only_ids": [], "metadata_only_ids": [],
                 "duplicate_matrix_ids": 0, "duplicate_metadata_ids": 0,
                 "metadata_rows_reordered": sum(metadata_positions[value] != i for i, value in enumerate(ids)),
                 "position_basis": "zero-based source positions; matrix positions refer to samples in the declared orientation",
                 "samples": [{"sample_id": value, "matrix_position": i,
                              "metadata_position": metadata_positions[value]} for i, value in enumerate(ids)]}
    if input_type == "counts":
        for column in values:
            for token in values[column]:
                try:
                    if allow_missing_counts and token in ("", "NA", "NaN", "nan", "N/A"):
                        continue
                    count = Decimal(token)
                    valid = count.is_finite() and count >= 0 and count == count.to_integral_value()
                except InvalidOperation:
                    valid = False
                if not valid:
                    raise ValueError(f"Raw counts must be finite, nonnegative integers: feature {column!r}, value {token!r}")
    values = values.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan, "N/A": np.nan})
    for column in values:
        try:
            values[column] = pd.to_numeric(values[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Nonnumeric feature values in {column!r}") from exc
    array = values.to_numpy(dtype=float)
    if input_type == "counts":
        observed = array[np.isfinite(array)]
        if np.isinf(array).any() or (not allow_missing_counts and np.isnan(array).any()) or (observed < 0).any() or (observed != np.floor(observed)).any():
            raise ValueError("Raw counts must be finite, nonnegative integers; missing counts are invalid")
        zero_ids = ids[(np.nan_to_num(array, nan=0) == 0).all(axis=1)].tolist()
        if zero_ids:
            raise ValueError(f"Zero library size for samples: {zero_ids}")
    elif np.isinf(array).any():
        raise ValueError("Numerical features must be finite or missing; infinity is invalid")
    values.insert(0, "sample_id", ids.to_numpy())
    metadata = metadata.set_index(sample_id_column).loc[ids].reset_index()
    rename = {sample_id_column: "sample_id", outcome_column: "outcome"}
    if any(target in metadata and source != target for source, target in rename.items()):
        raise ValueError("Metadata contains a column conflicting with canonical sample_id/outcome names")
    metadata = metadata.rename(columns=rename)
    configuration = {"orientation": orientation, "sample_id_column": sample_id_column,
                     "outcome_column": outcome_column, "feature_id_column": feature_id_column,
                     "positive_label": positive_label, "input_type": input_type,
                     "group_column": group_column, "cohort_column": cohort_column,
                     "allow_missing_counts": allow_missing_counts, "metadata_mapping": metadata_mapping}
    alignment["metadata_harmonization"] = harmonization
    return LoadedData(values, metadata, positive_label, input_type, group_column, cohort_column,
                      sources, alignment, configuration)
