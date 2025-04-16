"""Generate error tolerance tables for unspecified
compiler/platform/os combinations by creating tables that are looser
than all platform specific tables for a given collection of test cases
for a particular function for a particular set of input types.."""


import argparse
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from functools import reduce
from pathlib import Path

from xsref.tables import _get_git_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_path_root")
    parser.add_argument("--factor", type=float, default=4.0)
    args = parser.parse_args()
    table_path_root = Path(args.table_path_root)

    for inpath in table_path_root.glob("**/In_*.parquet"):
        types = inpath.name.removesuffix(".parquet").replace("In_", "")
        pattern = f"Err_{types}_*.parquet"
        other_err_table_name = pattern.replace("*", "other")
        err_tables = []
        for errpath in inpath.parent.glob(pattern):
            if errpath.name == other_err_table_name:
                continue
            err_tables.append(pl.read_parquet(errpath).to_numpy())
        other_err_table = reduce(lambda x, y: np.maximum(x, y), err_tables)
        factor = other_err_table.dtype.type(args.factor)
        with np.errstate(over="ignore"):
            other_err_table = np.maximum(
                other_err_table, np.finfo(other_err_table.dtype).eps
            ) * factor
        X = [
            pa.array(other_err_table[:, i]) for i in range(other_err_table.shape[1])
        ]
        other_err_table = pa.table(X, names=pq.read_schema(errpath).names)

        metadata = pq.read_schema(inpath).metadata
        metadata[b"input_checksum"] = b"NA"
        metadata[b"output_checksum"] = b"NA"
        metadata[b"scipy_version"] = b"NA"
        metadata[b"cpp_compiler"] = b"NA"
        metadata[b"cpp_compiler_version"] = b"NA"
        metadata[b"architecture"] = b"NA"
        metadata[b"operating_system"] = b"NA"
        commit_hash, working_tree = _get_git_info()
        metadata[b"xsref_commit_hash"] = commit_hash
        metadata[b"working_tree_state"] = working_tree

        other_err_table = other_err_table.replace_schema_metadata(metadata)
        outpath = inpath.parent / other_err_table_name
        pq.write_table(
            other_err_table, outpath, compression="zstd", compression_level=22
        )
        

            
