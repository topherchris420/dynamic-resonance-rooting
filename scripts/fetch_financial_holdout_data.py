#!/usr/bin/env python3
"""Fetch the registered daily bars; verify upstream checksums before normalization."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/financial_holdout"


def fetch_month(item):
    symbol, month, template = item
    url = template.format(symbol=symbol, month=month)
    with urlopen(url + ".CHECKSUM", timeout=60) as response:
        expected = response.read().decode().split()[0]
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"Upstream checksum mismatch: {url}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != 1:
            raise ValueError(f"Expected one CSV: {url}")
        raw = archive.read(archive.namelist()[0]).decode()
    rows = []
    for fields in csv.reader(io.StringIO(raw)):
        if len(fields) != 12:
            raise ValueError(f"Bad kline schema: {url}")
        start, end = int(fields[0]), int(fields[6])
        unit = 1_000_000 if start >= 10**15 else 1000
        if start % (86400 * unit) or end != start + 86400 * unit - 1:
            raise ValueError(f"Not a complete UTC daily bar: {url}")
        date = datetime.fromtimestamp(start // unit, timezone.utc).date().isoformat()
        rows.append([date, symbol, *fields[1:6]])
    year, number = map(int, month.split("-"))
    expected_dates = [
        f"{month}-{day:02d}" for day in range(1, calendar.monthrange(year, number)[1] + 1)
    ]
    if [row[0] for row in rows] != expected_dates:
        raise ValueError(f"Missing, unsorted or duplicate dates: {url}")
    return rows, {"url": url, "sha256": actual, "bytes": len(payload), "rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY / "data")
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output already exists; use a new directory, never overwrite a vintage.")
    protocol = json.loads((STUDY / "protocol.json").read_text())
    spec = protocol["data"]
    start_year, start_month = map(int, spec["start"][:7].split("-"))
    last_year, last_month = map(int, spec["end"][:7].split("-"))
    months = []
    year, month = start_year, start_month
    while (year, month) <= (last_year, last_month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    jobs = [(s, m, spec["source_template"]) for s in spec["symbols"] for m in months]
    rows, sources = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for monthly, source in pool.map(fetch_month, jobs):
            rows.extend(row for row in monthly if spec["start"] <= row[0] <= spec["end"])
            sources.append(source)
    rows.sort(key=lambda row: (row[0], row[1]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    manifest = {
        "provider": spec["provider"],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": spec["license"],
        "normalization": "UTC date and original decimal OHLCV strings; columns 0..5; filter registered range; sort date/symbol; LF CSV.",
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sources": sources,
    }
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())
        if manifest["sha256"] != expected["sha256"] or sources != expected["sources"]:
            raise SystemExit(
                "Source vintage changed; do not substitute data in the registered run."
            )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "bars.csv").write_bytes(payload)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"], "sources": len(sources)}))


if __name__ == "__main__":
    main()
