"""Measured environment locks and exact-result verification, not universal claims."""
from functools import wraps
import hashlib
from importlib.metadata import version, distribution
import json
import platform
from pathlib import Path
import sys
import shutil
import subprocess

from threadpoolctl import threadpool_info, threadpool_limits
from packaging.requirements import Requirement


def dependency_versions():
    pending = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "packaging"]
    pinned = {}
    while pending:
        name = pending.pop()
        key = name.lower().replace("_", "-")
        if key in pinned:
            continue
        pinned[key] = version(name)
        for text in distribution(name).requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return dict(sorted(pinned.items()))


def source_revision():
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"revision": None, "working_tree_dirty": None, "identity": "package implementation hashes"}
    git = shutil.which("git")
    if git is None and Path("C:/Program Files/Git/cmd/git.exe").is_file():
        git = "C:/Program Files/Git/cmd/git.exe"
    if git is None:
        return {"revision": None, "working_tree_dirty": None, "identity": "package implementation hashes"}
    try:
        revision = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output([git, "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL).strip())
        return {"revision": revision, "working_tree_dirty": dirty, "identity": "revision plus package implementation hashes"}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "working_tree_dirty": None, "identity": "package implementation hashes"}


def single_thread(function):
    @wraps(function)
    def controlled(*args, **kwargs):
        with threadpool_limits(limits=1):
            return function(*args, **kwargs)
    return controlled


def environment():
    root = Path(__file__).parent
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "packages": dependency_versions(),
            "package_code_sha256": code,
            "python_executable_sha256": hashlib.sha256(Path(getattr(sys, "_base_executable", sys.executable)).read_bytes()).hexdigest(),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads", "architecture")}
                            for p in threadpool_info()], "numerical_threads": 1}


def verify_environment(expected, actual):
    if expected is None:
        return
    if isinstance(expected, (str, Path)):
        expected = json.loads(Path(expected).read_text(encoding="utf-8"))
    expected = expected.get("reproducibility", {}).get("environment", expected)
    if expected != actual:
        differences = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise ValueError(f"Environment lock mismatch ({', '.join(differences)}); refusing to claim an exact reproduction")


def replay(run_path, features_path, metadata_path, output):
    """Recompute a saved run and publish only if environment/input/result hashes match."""
    from .benchmark import benchmark
    from .io import load_data
    from .engine import _fingerprint
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    if "reproducibility" not in run or "result_checksums" not in run:
        raise ValueError("This run predates verified replay; generate a new benchmark run first")
    configuration = run["configuration"]
    loading = dict(configuration["loading"])
    if not loading:
        raise ValueError("Replay requires a run with explicit file-loading configuration")
    data = load_data(features_path, metadata_path, **loading)
    f = data.features.sort_values("sample_id").reset_index(drop=True)
    m = data.metadata.set_index("sample_id").loc[f.sample_id].reset_index()
    if _fingerprint(f) != run["features_sha256"] or _fingerprint(m) != run["metadata_sha256"]:
        raise ValueError("Replay input checksum mismatch")
    keys = ("split", "holdout_cohort", "seed", "dataset_label", "normalization", "missing_counts",
            "batch_column", "weighting", "bootstrap_repeats")
    options = {key: configuration[key] for key in keys}
    for key in ("models", "tune", "inner_splits", "search_space", "time_column", "time_cutoff", "temporal_gap_days"):
        if key in configuration:
            options[key] = configuration[key]
    options["n_splits"] = configuration["n_splits_requested"]
    options["environment_lock"] = run["reproducibility"]["environment"]
    result = benchmark(data, **options)
    if result.provenance["result_checksums"] != run["result_checksums"]:
        raise ValueError("Exact replay result checksum mismatch; no output was published")
    result.provenance["reproducibility"]["verified_against_run"] = str(run_path)
    result.write(output)
    return result
