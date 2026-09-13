"""Reproducible, hashable analysis manifests; hashes are not digital signatures."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .common import canonical_json, stable_id, write_immutable


@dataclass(frozen=True)
class AnalysisPassport:
    payload_json: str

    def __post_init__(self):
        value = json.loads(self.payload_json)
        required = {
            "software_version",
            "git_commit",
            "timestamp",
            "as_of",
            "filing_vintages",
            "mapping_version",
            "institutions",
            "peer_definition",
            "variables",
            "transformations",
            "drr_configuration",
            "baseline_configuration",
            "robustness_configuration",
            "random_seeds",
            "source_hashes",
            "output_hashes",
        }
        if required - set(value):
            raise ValueError("Incomplete analysis passport")
        object.__setattr__(self, "payload_json", canonical_json(value))

    @property
    def analysis_id(self):
        return stable_id(json.loads(self.payload_json))

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @classmethod
    def create(cls, **payload):
        return cls(canonical_json(payload))

    def export(self, directory):
        return write_immutable(
            Path(directory) / f"passport-{self.analysis_id}.json",
            canonical_json(dict(analysis_id=self.analysis_id, **self.payload)) + "\n",
        )


def software_identity():
    try:
        version = importlib.metadata.version("drr-framework")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    try:
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(source.relative_to(package).as_posix().encode())
        digest.update(source.read_bytes())
    return {
        "version": version,
        "commit": commit,
        "working_tree_modified": dirty,
        "source_code_sha256": digest.hexdigest(),
    }


def dependency_inventory():
    """Installed-distribution inventory for reproducibility and external scanners."""
    return tuple(
        sorted(
            (d.metadata.get("Name", "unknown"), d.version)
            for d in importlib.metadata.distributions()
        )
    )
