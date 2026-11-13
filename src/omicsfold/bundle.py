"""Preserve the exact inputs, implementation and pinned runtime for a saved run."""
import hashlib
import json
from pathlib import Path
import zipfile
from importlib.metadata import distribution

from .report import OUTPUT_FILES
from .reproducibility import environment, verify_environment, single_thread


@single_thread
def bundle_run(run_path, features_path, metadata_path, output):
    run_path, output = Path(run_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Bundle destination already exists: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Bundle output must have a .zip extension")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if "reproducibility" not in run:
        raise ValueError("This run predates reproducibility bundles")
    verify_environment(run["reproducibility"]["environment"], environment())
    payload = {}
    paths = {}
    for role, path in (("features", Path(features_path)), ("metadata", Path(metadata_path))):
        raw = path.read_bytes()
        expected = run["input_checksums"]["source_files"].get(role, {}).get("sha256")
        if expected != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Exact source-file checksum mismatch for {role}")
        name = f"inputs/{role}{path.suffix.lower()}"
        payload[name] = raw
        paths[role] = name
    package = Path(__file__).parent
    for name, expected in run["reproducibility"]["environment"]["package_code_sha256"].items():
        raw = (package / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Implementation changed since this run; cannot bundle different source as the original")
        payload[f"engine/src/omicsfold/{name}"] = raw
    for file in sorted((package / "data").glob("*")):
        if file.is_file():
            payload[f"engine/src/omicsfold/data/{file.name}"] = file.read_bytes()
    pinned = run["reproducibility"]["environment"]["packages"]
    payload["requirements.lock.txt"] = ("\n".join(f"{key}=={value}" for key, value in sorted(pinned.items())) + "\n").encode()
    for name in OUTPUT_FILES:
        raw = (run_path.parent / name).read_bytes()
        if name != "run.json" and run.get("artifact_checksums", {}).get(name) != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Saved artifact checksum mismatch: {name}")
        payload[f"original/{name}"] = raw
    license_path = package.parents[1] / "LICENSE"
    if not license_path.is_file():
        license_path = next((distribution("omicsfold").locate_file(p) for p in distribution("omicsfold").files or []
                             if p.name == "LICENSE" and distribution("omicsfold").locate_file(p).is_file()), None)
    if license_path is None or not license_path.is_file():
        raise ValueError("Cannot locate the package license for source redistribution")
    payload["engine/LICENSE"] = license_path.read_bytes()
    payload["engine/pyproject.toml"] = ("[build-system]\nrequires = ['setuptools>=77.0.3']\n"
        "build-backend = 'setuptools.build_meta'\n[project]\nname = 'omicsfold'\n"
        f"version = '{run['omicsfold']}'\nrequires-python = '>=3.10'\nlicense = 'MIT'\nlicense-files = ['LICENSE']\n"
        "[project.scripts]\nomicsfold = 'omicsfold.cli:main'\n"
        "[tool.setuptools.packages.find]\nwhere = ['src']\n"
        "[tool.setuptools.package-data]\nomicsfold = ['data/*.tsv', 'data/*.csv']\n").encode()
    payload["REPRODUCE.txt"] = ("Use a matching Python/OS/hardware environment, then in an isolated environment:\n"
        "python -m pip install -r requirements.lock.txt\npython -m pip install --no-deps ./engine\n"
        f"omicsfold replay --run original/run.json --features {paths['features']} --metadata {paths['metadata']} --output verified-replay\n"
        "The replay gate rejects incompatible environments or different results. Pins do not guarantee package availability or portable bit identity.\n").encode()
    manifest = {"schema": 1, "python": run["python"], "source": run["reproducibility"].get("source"),
                "input_paths": paths, "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}}
    payload["bundle.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in payload.items():
            archive.writestr(name, raw)
    return output
