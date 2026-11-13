"""Install a built wheel into a clean venv and run it outside the checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tests", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    environment = work / "environment"
    venv.create(environment, with_pip=True, system_site_packages=False)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    cli = scripts / ("omicsfold.exe" if os.name == "nt" else "omicsfold")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    def run(*command):
        subprocess.run([str(x) for x in command], cwd=work, env=env, check=True)
    run(python, "-m", "pip", "install", str(wheel), *(["pytest"] if args.tests else []))
    run(python, "-c", "import importlib.metadata,omicsfold,sys; from pathlib import Path; "
        "assert importlib.metadata.version('omicsfold') == omicsfold.__version__ == '0.1.0'; "
        "assert Path(omicsfold.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "print('Installed wheel:',omicsfold.__file__)")
    run(cli, "example", "--output", "example")
    run(python, "-c", "import json; from pathlib import Path; p=Path('example'); "
        "assert set(x.name for x in p.iterdir()) == {'report.html','predictions.tsv','metrics.tsv','splits.tsv','features.tsv','run.json'}; "
        "r=json.loads((p/'run.json').read_text()); assert r['report_schema_version']=='1.0'; "
        "assert len(r['folds'][0]['train_ids'])==len(r['folds'][0]['test_ids'])==4; print('Outside-source example passed')")
    if args.tests:
        run(python, "-m", "pytest", str(args.tests.resolve()))
    (work / "installation.json").write_text(json.dumps({"wheel": str(wheel), "python": str(python),
        "cli": str(cli), "report": str(work / "example/report.html"), "system_site_packages": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
