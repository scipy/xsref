#!/usr/bin/env python3
"""List Windows tolerance parquet files that differ from baseline tables."""

from __future__ import annotations

import argparse
import math
from typing import Any
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BASE_PATH = Path(__file__).resolve().parents[1] / "tables" / "scipy_special_tests"
BASELINE = "other"
WINDOWS = "msvc-windows-x86_64"


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def format_value(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.17g}"
    return repr(value)


def format_series(row: pd.Series, max_items: int | None = None) -> str:
    items = list(row.items())
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(f"{name}={format_value(value)}" for name, value in items)


def changed_cells(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if list(left.columns) != list(right.columns):
        raise ValueError(f"column mismatch: {list(left.columns)} != {list(right.columns)}")

    equal = left.eq(right) | (left.isna() & right.isna())
    return ~equal


def signature_from_err_path(path: Path, platform: str) -> str:
    return path.name.removeprefix("Err_").removesuffix(f"_{platform}.parquet")


def context_paths(err_path: Path, signature: str) -> tuple[Path, Path]:
    return (
        err_path.with_name(f"In_{signature}.parquet"),
        err_path.with_name(f"Out_{signature}.parquet"),
    )


def print_changed_values(
    old_err: pd.DataFrame,
    new_err: pd.DataFrame,
    mask: pd.DataFrame,
    in_path: Path,
    out_path: Path,
    max_cells: int | None,
) -> None:
    input_df = read_parquet(in_path) if in_path.exists() else None
    output_df = read_parquet(out_path) if out_path.exists() else None
    printed = 0

    for row_idx, row_mask in mask.iterrows():
        changed_columns = [column for column, changed in row_mask.items() if changed]
        if not changed_columns:
            continue

        print(f"  row {row_idx}:")
        if input_df is not None:
            print(f"    input:  {format_series(input_df.loc[row_idx])}")
        if output_df is not None:
            print(f"    output: {format_series(output_df.loc[row_idx])}")

        for column in changed_columns:
            print(
                "    "
                f"{column}: {BASELINE}={format_value(old_err.at[row_idx, column])} "
                f"{WINDOWS}={format_value(new_err.at[row_idx, column])}"
            )
            printed += 1
            if max_cells is not None and printed >= max_cells:
                remaining = int(mask.to_numpy().sum()) - printed
                if remaining:
                    print(f"    ... truncated {remaining} changed cells")
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Maximum changed cells to print per file. Default: print all.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print changed files and counts, not row values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compared = 0
    changed_files = 0
    missing_windows = 0

    for old_path in sorted(BASE_PATH.glob(f"**/Err_*_{BASELINE}.parquet")):
        signature = signature_from_err_path(old_path, BASELINE)
        new_path = old_path.with_name(f"Err_{signature}_{WINDOWS}.parquet")

        if not new_path.exists():
            missing_windows += 1
            continue

        old_err = read_parquet(old_path)
        new_err = read_parquet(new_path)
        mask = changed_cells(old_err, new_err)

        compared += 1
        if not mask.to_numpy().any():
            continue

        changed_files += 1
        changed_rows = mask.any(axis=1)
        changed_cols = list(mask.columns[mask.any(axis=0)])
        rel_path = new_path.relative_to(BASE_PATH)

        print(f"{rel_path}")
        print(f"  changed rows: {int(changed_rows.sum())}")
        print(f"  changed cells: {int(mask.to_numpy().sum())}")
        print(f"  changed columns: {', '.join(changed_cols)}")

        in_path, out_path = context_paths(old_path, signature)
        if in_path.exists() and out_path.exists():
            print(f"  input table: {in_path.relative_to(BASE_PATH)}")
            print(f"  output table: {out_path.relative_to(BASE_PATH)}")

        if not args.summary_only:
            print_changed_values(
                old_err,
                new_err,
                mask,
                in_path,
                out_path,
                args.max_cells,
            )

        print()

    print(f"Compared {compared} Windows/baseline table pairs.")
    print(f"Found {changed_files} files with value diffs.")
    if missing_windows:
        print(f"Skipped {missing_windows} baseline files with no Windows table.")


if __name__ == "__main__":
    main()
