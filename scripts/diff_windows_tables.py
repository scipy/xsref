#!/usr/bin/env python3
"""Summarize MSVC Windows tolerance-table diffs.

Compares Err_*_msvc-windows-x86_64.parquet files with the matching
Err_*_other.parquet files and prints a Markdown report. For changed cells, the
report includes the row number plus matching In_* and Out_* row context.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_PLATFORM = "msvc-windows-x86_64"
DEFAULT_BASELINE = "other"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_tables_root() -> Path:
    return repo_root() / "tables" / "scipy_special_tests"


def read_columns(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.schema.names
    }


def read_row(path: Path | None, row: int) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    table = pq.read_table(path)
    return {name: table[name][row].as_py() for name in table.schema.names}


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def changed_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")

    if np.issubdtype(left.dtype, np.floating) and np.issubdtype(right.dtype, np.floating):
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        return ~same

    return np.array(
        [not values_equal(left_value, right_value) for left_value, right_value in zip(left, right)],
        dtype=bool,
    )


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.17g}"
    if isinstance(value, complex):
        return f"{value.real:.17g}{value.imag:+.17g}j"
    return repr(value)


def format_row(row: dict[str, Any]) -> str:
    if not row:
        return ""
    return ", ".join(f"{key}={format_value(value)}" for key, value in row.items())


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def parse_signature(path: Path, platform: str) -> str:
    prefix = "Err_"
    suffix = f"_{platform}.parquet"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"unexpected table name: {path}")
    return path.name[len(prefix):-len(suffix)]


def metadata_diff(left_path: Path, right_path: Path) -> list[tuple[str, str, str]]:
    left = pq.read_schema(left_path).metadata or {}
    right = pq.read_schema(right_path).metadata or {}
    keys = sorted(set(left) | set(right))
    diffs = []
    for key in keys:
        left_value = left.get(key, b"").decode("utf-8", errors="replace")
        right_value = right.get(key, b"").decode("utf-8", errors="replace")
        if left_value != right_value:
            diffs.append((key.decode("utf-8", errors="replace"), left_value, right_value))
    return diffs


def build_report(args: argparse.Namespace) -> str:
    tables_root = args.tables_root.resolve()
    pattern = f"**/Err_*_{args.platform}.parquet"
    windows_tables = sorted(tables_root.glob(pattern))

    total_tables = 0
    changed_tables = 0
    total_changed_cells = 0
    total_changed_rows = 0
    sections: list[str] = []
    missing_baselines: list[Path] = []

    for windows_path in windows_tables:
        signature = parse_signature(windows_path, args.platform)
        baseline_path = windows_path.with_name(f"Err_{signature}_{args.baseline}.parquet")
        input_path = windows_path.with_name(f"In_{signature}.parquet")
        output_path = windows_path.with_name(f"Out_{signature}.parquet")

        if not baseline_path.exists():
            missing_baselines.append(windows_path)
            continue

        total_tables += 1
        windows_cols = read_columns(windows_path)
        baseline_cols = read_columns(baseline_path)
        all_columns = sorted(set(windows_cols) | set(baseline_cols))

        diffs: list[tuple[int, str, Any, Any]] = []
        changed_rows: set[int] = set()
        for name in all_columns:
            if name not in windows_cols or name not in baseline_cols:
                raise ValueError(f"column mismatch in {windows_path}: {name}")
            mask = changed_mask(windows_cols[name], baseline_cols[name])
            rows = np.flatnonzero(mask)
            for row in rows:
                diffs.append((int(row), name, baseline_cols[name][row], windows_cols[name][row]))
                changed_rows.add(int(row))

        meta_diffs = metadata_diff(baseline_path, windows_path) if args.metadata else []
        if not diffs and not meta_diffs and args.changed_only:
            continue

        if diffs or meta_diffs:
            changed_tables += 1
        total_changed_cells += len(diffs)
        total_changed_rows += len(changed_rows)

        function_name = windows_path.parent.name
        title = f"### `{function_name}` / `{signature}`"
        lines = [
            title,
            "",
            f"- file: `{windows_path.relative_to(tables_root)}`",
            f"- changed tolerance cells: `{len(diffs)}`",
            f"- changed rows: `{len(changed_rows)}`",
        ]

        if meta_diffs:
            lines.extend(["", "Metadata diffs:", "", "| key | baseline | windows |", "| --- | --- | --- |"])
            for key, baseline_value, windows_value in meta_diffs:
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (key, baseline_value, windows_value)
                    )
                    + " |"
                )

        if diffs:
            lines.extend(
                [
                    "",
                    "| row | column | baseline | windows | input row | output row |",
                    "| ---: | --- | ---: | ---: | --- | --- |",
                ]
            )
            for row, column, baseline_value, windows_value in diffs[: args.max_cells]:
                input_row = format_row(read_row(input_path, row))
                output_row = format_row(read_row(output_path, row))
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            str(row),
                            f"`{column}`",
                            format_value(baseline_value),
                            format_value(windows_value),
                            input_row,
                            output_row,
                        )
                    )
                    + " |"
                )
            if len(diffs) > args.max_cells:
                lines.append("")
                lines.append(f"_Truncated: showing {args.max_cells} of {len(diffs)} changed cells._")

        sections.append("\n".join(lines))

    summary = [
        "# Windows Parquet Diff Report",
        "",
        f"- tables root: `{tables_root}`",
        f"- platform: `{args.platform}`",
        f"- baseline: `{args.baseline}`",
        f"- tables compared: `{total_tables}`",
        f"- tables with diffs: `{changed_tables}`",
        f"- changed tolerance cells: `{total_changed_cells}`",
        f"- changed rows, summed per table: `{total_changed_rows}`",
    ]
    if missing_baselines:
        summary.append(f"- missing baseline tables: `{len(missing_baselines)}`")

    if missing_baselines and args.show_missing:
        summary.extend(["", "Missing baseline tables:", ""])
        summary.extend(f"- `{path.relative_to(tables_root)}`" for path in missing_baselines)

    return "\n".join(summary + [""] + sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-root",
        type=Path,
        default=default_tables_root(),
        help="Root containing scipy_special_tests table directories.",
    )
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=200,
        help="Maximum changed cells to show per table.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Also show parquet metadata differences.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="List Windows tables with no matching baseline table.",
    )
    parser.add_argument(
        "--all",
        dest="changed_only",
        action="store_false",
        help="Include unchanged tables in the report.",
    )
    parser.set_defaults(changed_only=True)
    return parser.parse_args()


def main() -> None:
    print(build_report(parse_args()))


if __name__ == "__main__":
    main()
