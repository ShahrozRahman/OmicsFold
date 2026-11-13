"""Check both distribution versions and the source/fixture payload; no publishing."""
import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile


def check(directory, version):
    directory = Path(directory)
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution in a fresh directory")
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        message = BytesParser().parsebytes(archive.read(metadata))
        assert message["Name"] == "omicsfold" and message["Version"] == version
        assert f"omicsfold-{version}-" in wheels[0].name
        for name in ("omicsfold/data/counts.tsv", "omicsfold/data/metadata.csv"):
            assert name in archive.namelist(), name
    with tarfile.open(sources[0]) as archive:
        root = f"omicsfold-{version}"
        metadata = archive.extractfile(f"{root}/PKG-INFO")
        message = BytesParser().parsebytes(metadata.read())
        assert message["Name"] == "omicsfold" and message["Version"] == version
        assert sources[0].name == f"{root}.tar.gz"
        for name in ("pyproject.toml", "examples/benchmark/counts.tsv", "r/omicsfold/DESCRIPTION",
                     "r/omicsfold/tests/benchmark.R", "tests/test_report.py"):
            assert f"{root}/{name}" in archive.getnames(), name
    print(f"Verified wheel and sdist metadata/payload: omicsfold {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    check(args.directory, args.version)
