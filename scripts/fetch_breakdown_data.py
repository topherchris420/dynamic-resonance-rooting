#!/usr/bin/env python3
"""Download and pin the registered hourly market-data vintage."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    symbol, month = item
    url = (
        f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    )
    with urlopen(url + ".CHECKSUM", timeout=60) as response:
        expected = response.read().decode().split()[0]
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Upstream checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != 1:
            raise ValueError("Unexpected archive schema")
        content = archive.read(archive.namelist()[0]).decode()
    rows = []
    for fields in csv.reader(io.StringIO(content)):
        if len(fields) != 12:
            raise ValueError("Unexpected kline schema")
        start, end = int(fields[0]), int(fields[6])
        unit = 1000000 if start >= 10**15 else 1000
        if start % (3600 * unit) or end != start + 3600 * unit - 1:
            raise ValueError("Invalid hourly bar boundary")
        date = datetime.fromtimestamp(start // unit, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        rows.append([date, symbol, *fields[1:6]])
    return rows, {"url": url, "sha256": expected, "bytes": len(payload), "rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Use a new output directory")
    protocol = json.loads((ROOT / "experiments/lead_lag_breakdown/protocol.json").read_text())
    jobs = [(s, m) for s in protocol["market"]["symbols"] for m in protocol["market"]["months"]]
    rows, sources = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for values, source in pool.map(fetch, jobs):
            rows.extend(values)
            sources.append(source)
    rows.sort(key=lambda r: (r[0], r[1]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["time", "symbol", "open", "high", "low", "close", "volume"])
    writer.writerows(rows)
    payload = gzip.compress(stream.getvalue().encode(), mtime=0)
    manifest = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "sources": sources,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Binance Vision, CC BY-NC-SA 4.0 and dataset terms",
        "normalization": "Original OHLCV strings, UTC hourly timestamps, sorted time/symbol, LF CSV, deterministic gzip",
    }
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())
        if expected["sha256"] != manifest["sha256"] or expected["sources"] != sources:
            raise ValueError("Archive vintage changed")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "bars.csv.gz").write_bytes(payload)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "archives": len(sources)}))


if __name__ == "__main__":
    main()
