"""Training-fitted normalization and batch adjustment for held-out evaluation."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class AssayPreprocessor:
    def __init__(self, input_type, normalization="auto", missing_counts="error"):
        self.input_type = input_type
        self.normalization = ("cpm_log1p" if input_type == "counts" else "none") if normalization == "auto" else normalization
        self.missing_counts = missing_counts
        if self.normalization not in ("none", "cpm_log1p", "median_ratio_log1p", "log1p", "clr"):
            raise ValueError("Unknown normalization")
        if input_type != "counts" and self.normalization in ("cpm_log1p", "median_ratio_log1p"):
            raise ValueError("Library/size-factor normalization requires counts")
        if missing_counts not in ("error", "median"):
            raise ValueError("missing_counts must be error or median")

    def _complete(self, X):
        if self.input_type == "counts" and self.missing_counts == "error":
            if not np.isfinite(X).all():
                raise ValueError("Missing counts require explicit missing_counts='median'")
            return X
        return self.imputer.transform(X[:, self.observed])

    def _normalized(self, X):
        if self.normalization == "cpm_log1p":
            maxima = X.max(axis=1, keepdims=True)
            if (maxima <= 0).any():
                raise ValueError("Zero library size after count imputation")
            relative = X / maxima
            return np.log1p(relative / relative.sum(axis=1, keepdims=True) * 1e6)
        if self.normalization == "median_ratio_log1p":
            ratios = X[:, self.reference_features] / self.reference
            positive = ratios > 0
            factors = np.array([np.median(row[ok]) if ok.any() else np.nan for row, ok in zip(ratios, positive)]) / self.factor_center
            if not np.isfinite(factors).all() or (factors <= 0).any():
                raise ValueError("Cannot estimate size factor: sample has no positive reference counts")
            return np.log1p(X / factors[:, None])
        if self.normalization in ("log1p", "clr"):
            if (X < 0).any():
                raise ValueError(f"{self.normalization} requires nonnegative feature values")
            transformed = np.log1p(X)
            return transformed - transformed.mean(axis=1, keepdims=True) if self.normalization == "clr" else transformed
        return X

    def fit(self, X, batches=None, outcomes=None):
        self.observed = np.isfinite(X).any(axis=0)
        if not self.observed.any():
            raise ValueError("Preprocessing leaves no usable features: all training values are missing")
        if self.input_type == "counts" and self.missing_counts == "error":
            completed = self._complete(X)
            self.observed[:] = True
        else:
            self.imputer = SimpleImputer(strategy="median").fit(X[:, self.observed])
            completed = self._complete(X)
        if self.normalization == "median_ratio_log1p":
            self.reference_features = (completed > 0).all(axis=0)
            if not self.reference_features.any():
                raise ValueError("No positive-in-all-training-samples reference genes for median-ratio normalization")
            self.reference = np.exp(np.log(completed[:, self.reference_features]).mean(axis=0))
            raw_factors = np.median(completed[:, self.reference_features] / self.reference, axis=1)
            self.factor_center = float(np.exp(np.log(raw_factors).mean()))
        transformed = self._normalized(completed)
        self.mask = self.observed.copy()
        if self.input_type == "counts":
            maxima = completed.max(axis=1, keepdims=True)
            if (maxima <= 0).any():
                raise ValueError("Zero library size after count imputation")
            relative = completed / maxima
            cpm = relative / relative.sum(axis=1, keepdims=True) * 1e6
            expressed = (cpm >= 1).sum(axis=0) >= int(np.ceil(.1 * len(X)))
            self.mask[self.observed] = expressed
            transformed = transformed[:, expressed]
        if not self.mask.any():
            raise ValueError("Preprocessing leaves no usable features after training expression filtering")
        self.batch_offsets = None
        if batches is not None:
            if outcomes is None:
                raise ValueError("Batch adjustment requires training outcomes for confounding checks")
            for batch in np.unique(batches):
                if np.unique(outcomes[batches == batch]).size != 2:
                    raise ValueError(f"Batch {batch!r} lacks both training outcomes; cannot separate batch from phenotype")
            levels = np.unique(batches)
            design = np.column_stack([np.ones(len(X)), outcomes] + [(batches == batch).astype(float) for batch in levels[1:]])
            if np.linalg.matrix_rank(design) != design.shape[1]:
                raise ValueError("Batch and phenotype design is not identifiable")
            coefficients = np.linalg.lstsq(design, transformed, rcond=None)[0]
            offsets = {str(levels[0]): np.zeros(transformed.shape[1])}
            offsets.update({str(batch): coefficients[i + 2] for i, batch in enumerate(levels[1:])})
            center = np.array([offsets[str(batch)] for batch in batches]).mean(axis=0)
            self.batch_offsets = {batch: offset - center for batch, offset in offsets.items()}
            transformed = self._adjust(transformed, batches)
        self.variable = np.ptp(transformed, axis=0) > 0
        if not self.variable.any():
            raise ValueError("Preprocessing leaves no usable features: training features are constant")
        self.selected_indices = np.flatnonzero(self.mask)[self.variable]
        self.scaler = StandardScaler().fit(transformed[:, self.variable])
        return self

    def _adjust(self, transformed, batches):
        if self.batch_offsets is None:
            return transformed
        if batches is None:
            raise ValueError("Batch labels are required to transform samples")
        unseen = sorted(set(map(str, batches)) - set(self.batch_offsets))
        if unseen:
            raise ValueError(f"Unseen test batches cannot be corrected without learning from test samples: {unseen}")
        return transformed - np.array([self.batch_offsets[str(batch)] for batch in batches])

    def transform(self, X, batches=None):
        completed = self._complete(X)
        transformed = self._normalized(completed)
        transformed = transformed[:, self.mask[self.observed]]
        return self.scaler.transform(self._adjust(transformed, batches)[:, self.variable])

    def state(self):
        return {"normalization": self.normalization, "missing_counts": self.missing_counts,
                "observed_features": self.observed.tolist(), "retained_indices": self.selected_indices.tolist(),
                "imputation_medians": self.imputer.statistics_.tolist() if hasattr(self, "imputer") else None,
                "size_factor_reference": self.reference.tolist() if hasattr(self, "reference") else None,
                "size_factor_center": self.factor_center if hasattr(self, "factor_center") else None,
                "batch_offsets": {k: v.tolist() for k, v in self.batch_offsets.items()} if self.batch_offsets else None,
                "scaling_mean": self.scaler.mean_.tolist(), "scaling_scale": self.scaler.scale_.tolist()}
