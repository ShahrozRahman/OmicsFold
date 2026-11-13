"""Versioned, self-contained benchmark reports with escaped user text."""
import html
import json
from pathlib import Path
import tempfile
import hashlib

import pandas as pd
from sklearn.metrics import precision_recall_curve


REPORT_SCHEMA_VERSION = "1.0"
OUTPUT_FILES = ("report.html", "predictions.tsv", "metrics.tsv", "splits.tsv", "features.tsv", "run.json")


def metric_rows(result):
    """Per-fold rows followed by unweighted arithmetic fold means."""
    per_fold = result.metrics.copy()
    per_fold.insert(0, "scope", "fold")
    numeric = ["average_precision", "roc_auc", "test_positive_prevalence", "test_sample_count"]
    means = result.metrics.groupby("model", sort=True)[numeric].mean().reset_index()
    means.insert(0, "fold", "")
    means.insert(0, "scope", "mean")
    return pd.concat([per_fold, means], ignore_index=True)


def _table(frame):
    return frame.to_html(index=False, escape=True, border=0, na_rep="", float_format=lambda x: f"{x:.6g}")


def _pr_curve(result):
    fold = result.folds[0]["fold"]
    label = (f"Named cohort holdout: {result.provenance['holdout_cohort']} (fold {fold})"
             if result.provenance["split"] == "cohort" else f"Individual fold {fold} only; not pooled across folds")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 400" role="img" aria-labelledby="pr-title">',
             f'<title id="pr-title">Precision–recall curve — {html.escape(label)}</title>',
             '<rect width="640" height="400" fill="white"/>']
    for tick in (0, .25, .5, .75, 1):
        x, y = 65 + 520 * tick, 320 - 270 * tick
        parts.append(f'<path d="M{x} 50V320 M65 {y}H585" stroke="#e2e8f0" fill="none"/>')
        parts.append(f'<text x="{x}" y="342" text-anchor="middle" font-size="13">{tick:g}</text>')
        parts.append(f'<text x="53" y="{y + 4}" text-anchor="end" font-size="13">{tick:g}</text>')
    parts.extend(['<path d="M65 50V320H585" fill="none" stroke="#334155"/>',
                  '<text x="325" y="374" text-anchor="middle">Recall</text>',
                  '<text x="18" y="185" transform="rotate(-90 18 185)" text-anchor="middle">Precision</text>'])
    fitted_label = next(name for name in result.predictions.model.unique() if name != "prevalence")
    for model, color, legend_y in (("prevalence", "#b45309", 18), (fitted_label, "#0369a1", 38)):
        samples = result.predictions[(result.predictions.fold == fold) & (result.predictions.model == model)]
        y = (samples.outcome == result.provenance["positive_label"]).astype(int)
        precision, recall, _ = precision_recall_curve(y, samples.positive_probability,
                                                     sample_weight=samples.get("evaluation_weight"))
        coordinates = list(zip(65 + 520 * recall, 320 - 270 * precision))
        x, y = coordinates[0]
        path = f"M{x:.5f} {y:.5f}" + "".join(f"H{x:.5f}V{y:.5f}" for x, y in coordinates[1:])
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="350" y="{legend_y}" fill="{color}" font-size="14">{html.escape(model)}</text>')
    parts.append("</svg>")
    return label, "".join(parts)


def render_report(result):
    """Render offline HTML: inline styles/SVG, no scripts or external resources."""
    run = result.provenance
    escape = lambda value: html.escape(str(value), quote=True)
    audit = run["sample_alignment_audit"]["source_alignment"]
    audit_summary = pd.DataFrame([{
        "matched_samples": audit["matched_samples"], "matrix_only": len(audit["matrix_only_ids"]),
        "metadata_only": len(audit["metadata_only_ids"]), "duplicate_matrix_ids": audit["duplicate_matrix_ids"],
        "duplicate_metadata_ids": audit["duplicate_metadata_ids"], "metadata_rows_reordered": audit["metadata_rows_reordered"]}])
    balance = pd.DataFrame([{"outcome": key, "samples": value} for key, value in run["class_balance"].items()])
    balance["fraction"] = balance.samples / balance.samples.sum()
    sizes = result.splits.groupby(["fold", "partition"], sort=True).agg(
        sample_count=("sample_id", "size"), positive_count=("outcome", lambda s: int((s == run["positive_label"]).sum()))).reset_index()
    sizes["positive_prevalence"] = sizes.positive_count / sizes.sample_count
    retention = result.features.groupby("fold", sort=True).agg(input_features=("feature", "size"),
                                                                retained_features=("retained", "sum")).reset_index()
    retention["dropped_features"] = retention.input_features - retention.retained_features
    rankings = result.features[result.features.retained & result.features.coefficient.notna()].sort_values(["fold", "absolute_coefficient_rank"]).groupby("fold", sort=True).head(20)
    importance_html = ""
    if "impurity_importance" in result.features:
        importance = result.features[result.features.impurity_importance.notna()].sort_values(["fold", "impurity_importance"], ascending=[True, False]).groupby("fold").head(20)
        importance_html = "<p>Random-forest impurity importances are exploratory, can favor correlated/high-cardinality features, and are not logistic coefficients.</p>" + _table(importance)
    design = run.get("validation_design", {})
    display_design = {key: value for key, value in design.items() if key not in ("scored_samples", "unscored_samples")}
    display_design["scored_sample_count"] = len(design.get("scored_samples", []))
    display_design["unscored_sample_count"] = len(design.get("unscored_samples", []))
    selection_rows = [{"outer_fold": f["fold"], "selected_candidate": str(f.get("selected_candidate")),
                       "inner_selection": f.get("selection") is not None} for f in result.folds]
    trial_rows = [{"outer_fold": f["fold"], "candidate": str(trial["candidate"]),
                   "mean_inner_AP": trial["mean_inner_average_precision"]}
                  for f in result.folds if f.get("selection") for trial in f["selection"]["candidates"]]
    trial_html = _table(pd.DataFrame(trial_rows)) if trial_rows else "<p>No inner model search was requested.</p>"
    curve_label, curve = _pr_curve(result)
    weighting = run["configuration"].get("weighting", "sample")
    interval_rows = []
    for fold in result.folds:
        for model, interval in fold.get("intervals", {}).items():
            if interval is not None:
                interval_rows.append({"fold": fold["fold"], "model": model,
                                      "AP lower": interval["average_precision"][0], "AP upper": interval["average_precision"][1],
                                      "AUC lower": interval["roc_auc"][0], "AUC upper": interval["roc_auc"][1],
                                      "valid cluster resamples": interval["valid_resamples"]})
    intervals_html = (_table(pd.DataFrame(interval_rows)) if interval_rows else "<p>Bootstrap intervals were not requested.</p>")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmicsFold — {escape(run['dataset_label'])}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#172033;background:#f8fafc;margin:0}}main{{max-width:1100px;margin:auto;padding:2rem}}section{{background:white;border:1px solid #cbd5e1;border-radius:8px;margin:1.5rem 0;padding:1.25rem;overflow:auto}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:.5rem;text-align:left;border-bottom:1px solid #e2e8f0}}th{{background:#f1f5f9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}svg{{max-width:720px;width:100%;height:auto}}.note{{color:#475569}}</style>
</head><body><main>
<h1>OmicsFold benchmark: {escape(run['dataset_label'])}</h1>
<p>Report schema {escape(REPORT_SCHEMA_VERSION)}. Positive label (user-specified): <strong>{escape(run['positive_label'])}</strong>.</p>
<p>Held-out evaluation weighting: <strong>{escape(weighting)}</strong>. Group-balanced mode gives each test group total weight one, regardless of sample count, and also weights the training loss by group. Scores remain available per sample; groups are not asserted to be statistically independent.</p>
<section><h2>Sample alignment audit</h2><p>Exact IDs; no intersection or silent dropping. Source metadata are reordered to matrix sample order, then benchmark samples are sorted by ID.</p>
{_table(audit_summary)}<p>{escape(audit['position_basis'])}. First 20 source matches below; run.json contains the full audit.</p>
{_table(pd.DataFrame(audit['samples']).head(20))}</section>
<section><h2>Class balance</h2>{_table(balance)}</section>
<section><h2>Split strategy and fold sizes</h2><p>Strategy: <strong>{escape(run['split'])}</strong>; seed: {escape(run['seed'])}; named holdout: {escape(run['holdout_cohort'])}.</p>
<p>Train/test IDs and any supplied groups were checked for overlap before fitting. Both partitions contain both outcomes. splits.tsv records all training and test memberships, including cohort and group labels when supplied.</p>{_table(sizes)}</section>
<section><h2>Model comparison: per-fold results and mean across folds</h2>
<p>The prevalence model predicts the training partition's positive prevalence. Mean rows are unweighted arithmetic means of the fold rows, not pooled scores; test sample count in a mean row is the mean fold size. A single holdout's mean equals that holdout's result.</p>
{_table(metric_rows(result))}</section>
<section><h2>Model selection and validation design</h2>
<p>Tuning occurs only in inner folds of the outer training partition. All learned preprocessing is refitted inside inner training folds, then refitted on the full outer training partition after selection. Outer test outcomes never select candidates. Equal inner scores choose the first predeclared candidate.</p>
{_table(pd.DataFrame(selection_rows))}{trial_html}<pre>{escape(json.dumps(display_design, indent=2))}</pre>
<p>Chronological validation trains strictly before the test window, with any configured embargo. Initial training and embargo/future rows may be unscored; their IDs are recorded. A named external cohort or temporal holdout provides evidence for that data, not a general proof of external validity. Repeatedly selecting settings from outer scores can still introduce selection bias.</p></section>
<section><h2>Cluster-aware uncertainty and reproducibility</h2>
<p>Intervals resample whole test groups with replacement, preserving within-group correlation. They condition on the fitted training model, assume the supplied groups are independent sampling units, and do not cover retraining uncertainty. No independence is manufactured by splitting or weighting. Fold intervals are not averaged.</p>{intervals_html}
<p>Numerical libraries are limited to one thread. run.json records the environment and implementation hashes, training preprocessing states and result checksums. Exact replay must verify all three; a changed platform, dependency, implementation or result is rejected rather than labeled identical.</p></section>
<section><h2>Precision–recall curve</h2><p>{escape(curve_label)}. Only held-out samples from this partition are plotted for both models.</p>{curve}</section>
<section><h2>Exploratory feature retention and coefficient rankings</h2>
<p>These are training-fold model outputs, not biological evidence, significance tests, validated biomarkers, or causal effects. Coefficients describe standardized transformed features for the user-specified positive label; correlated features can have unstable rankings.</p>
{_table(retention)}<p>Top 20 retained features per fold, ranked by absolute coefficient. Signed coefficients are shown. All retained and dropped features and reasons are in features.tsv; baseline has no feature coefficients.</p>
{_table(rankings)}{importance_html}<p>Logistic coefficients/intercepts are absent for non-logistic models. Model intercepts (standardized feature space): {escape({f['fold']: f['intercept'] for f in result.folds})}.</p></section>
<section><h2>Configuration, input checksums and package versions</h2>
<p>Source-file SHA-256 hashes refer to exact file bytes when loaded from files. Canonical hashes refer to the aligned input tables. Direct in-memory fixtures have canonical hashes only.</p>
<pre>{escape(json.dumps({'configuration': run['configuration'], 'input_checksums': run['input_checksums'], 'package_versions': run['package_versions']}, indent=2, allow_nan=False))}</pre></section>
</main></body></html>'''


def write_results(result, directory):
    """Stage complete files, then claim a new destination without overwriting."""
    path = Path(directory).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".omicsfold-", dir=path.parent) as temporary:
        staging = Path(temporary)
        result.predictions.to_csv(staging / "predictions.tsv", sep="\t", index=False)
        metrics = metric_rows(result)
        metrics.to_csv(staging / "metrics.tsv", sep="\t", index=False)
        result.splits.to_csv(staging / "splits.tsv", sep="\t", index=False)
        features = result.features.copy()
        features["absolute_coefficient_rank"] = features.absolute_coefficient_rank.astype("Int64")
        features.to_csv(staging / "features.tsv", sep="\t", index=False, na_rep="")
        (staging / "report.html").write_text(render_report(result), encoding="utf-8")
        run = dict(result.provenance, report_schema_version=REPORT_SCHEMA_VERSION,
                   output_files=list(OUTPUT_FILES), folds=result.folds,
                   metric_summary={"mean": "unweighted arithmetic mean across folds", "rows": metrics.to_dict(orient="records")})
        run["artifact_checksums"] = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                     for name in OUTPUT_FILES if name != "run.json"}
        (staging / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Exclusive directory creation also handles another writer claiming the
        # destination during staging. Existing destinations are never modified.
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"Output destination already exists: {path}") from exc
        published = []
        try:
            for name in OUTPUT_FILES:
                (staging / name).rename(path / name)
                published.append(path / name)
        except BaseException:
            # Remove only files this invocation published, never preexisting work.
            for file in published:
                file.unlink(missing_ok=True)
            path.rmdir()
            raise
