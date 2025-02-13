import argparse
from pathlib import Path
import pyarrow.parquet as pq

from xsref.tables import compute_initial_err_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_path_root")
    args = parser.parse_args()
    table_path_root = Path(args.table_path_root)

    for inpath in table_path_root.glob("**/In_*.parquet"):
        tol_table = compute_initial_err_table(inpath)
        metadata = tol_table.schema.metadata
        cpp_compiler = metadata[b"cpp_compiler"].decode("ascii")
        architecture = metadata[b"architecture"].decode("ascii")
        operating_system = metadata[b"operating_system"].decode("ascii")
        intypes = metadata[b"in"].decode("ascii")
        outtypes = metadata[b"out"].decode("ascii")

        filename = (
            f"Err_{intypes}_{outtypes}_"
            f"{cpp_compiler}-{operating_system}-{architecture}.parquet"
        )
        outpath = inpath.parent / filename
        pq.write_table(tol_table, outpath, compression="zstd", compression_level=22)
