#!/usr/bin/env python3
"""Compare reproduced CSV inputs with the reviewed paper tables.

Model-budget and convergence checks run separately in the experiment suite.
This comparison allows the rounding of older CSV tables and floating-point
differences between builds, while checking row counts, identifiers, and values.
"""
import argparse
import csv
import math
from pathlib import Path

from project_paths import analysis_dir, find_paper_dir, project_path


def read(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def number(value):
    if value.strip() == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tables",
        type=Path,
        default=analysis_dir / "tables",
        help="Fresh tables (default: analysis/vic-mf6-manuscript/tables)",
    )
    p.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Paper checkout (default: sibling vic-mf6-paper)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "table-comparison.csv",
        help="Comparison report (default: analysis/vic-mf6-manuscript/table-comparison.csv)",
    )
    a = p.parse_args()
    a.tables = project_path(a.tables)
    a.paper_dir = find_paper_dir() if a.paper_dir is None else project_path(a.paper_dir)
    a.output = project_path(a.output)
    records = []
    for path in sorted(a.tables.glob("data-*.csv")):
        previous = a.paper_dir / "manuscript" / path.name
        if not previous.exists():
            continue
        columns, new = read(path)
        old_columns, old = read(previous)
        # Local provenance paths are retained with raw runs, not plotting inputs.
        metadata = {"source_files", "vic_state_file"}
        expected = set(old_columns) - metadata
        passed = len(new) == len(old) and expected <= set(columns)
        records.append(dict(table=path.name, column="structure", count=len(new),
                            maximum_absolute_difference="", tolerance="same rows and required columns", passed=passed))
        if not passed:
            continue
        for key in sorted(expected):
            max_difference = 0.
            passed = True
            absolute = 1e-4 if ("error" in key or "residual" in key) and "m3" in key else 5e-6
            for left, right in zip(old, new):
                x, y = number(left[key]), number(right[key])
                if x is None or y is None:
                    passed &= left[key] == right[key]
                elif math.isnan(x) or math.isnan(y):
                    passed &= math.isnan(x) and math.isnan(y)
                elif not math.isfinite(x) or not math.isfinite(y):
                    passed &= x == y
                else:
                    difference = abs(x - y)
                    max_difference = max(max_difference, difference)
                    passed &= difference <= absolute + 1e-8 * max(abs(x), abs(y))
            records.append(dict(table=path.name, column=key, count=len(new),
                maximum_absolute_difference=max_difference, tolerance=f"{absolute:g} + 1e-8 * magnitude", passed=passed))
    if not records:
        raise ValueError("No corresponding paper tables found")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    failed = [row for row in records if not row["passed"]]
    print(f"Compared {len(set(row['table'] for row in records))} tables; {len(failed)} failed column/structure checks")
    for row in failed:
        print(row)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
