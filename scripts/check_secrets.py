"""Small, dependency-free, high-confidence secret gate for tracked text files.

This complements repository-host secret protection. It intentionally reports only the
location and detector name, never the matched value.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82,255}\b")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "literal credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)"
            r"\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
        ),
    ),
)


def tracked_files(root: Path) -> Tuple[Path, ...]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return tuple(root / name.decode("utf-8") for name in output.split(b"\0") if name)


def find_potential_secrets(paths: Iterable[Path], root: Path):
    findings = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            for detector, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append((path.relative_to(root).as_posix(), line_number, detector))
    return tuple(findings)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = find_potential_secrets(tracked_files(root), root)
    if findings:
        for path, line, detector in findings:
            print(f"{path}:{line}: potential {detector}")
        print("Potential secrets found; inspect and rotate/remove any real credential.")
        return 1
    print("No high-confidence secret patterns found in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
