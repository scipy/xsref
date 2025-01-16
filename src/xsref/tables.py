import numpy as np
import polars as pl
import pyarrow.parquet as pq


def _process_arg(arg, typecode):
    match typecode:
        case "d":
            return np.float64(arg)
        case "f":
            return np.float32(arg)
        case "D":
            return np.complex128(arg["real"], arg["imag"])
        case "F":
            return np.complex64(arg["real"], arg["imag"])
        case "i":
            return np.int32(arg)
        case "p":
            return np.intp(arg)
        case "_":
            raise ValueError(f"Received unhandled typecode: {typecode}")


def iter_inputs_table(inputs_table_path):
    """Iterate through test cases in inputs parquet table.

    Parameters
    ----------
    inputs_table_path : str
        Path to a parquet table with rows corresponding to arguments to
        a special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Complex inputs
        are stored as structs {"real": x, "imag": y} where x and y have the
        corresponding base type.
    Returns
    -------
    iterator of tuple
        Iterates through arguments which can be passed directly to a reference
        function.
    """
    metadata = pq.read_schema(inputs_table_path).metadata
    in_types = metadata[b"in"].decode("ascii")
    table = pl.read_parquet(inputs_table_path)
    for row in table.iter_rows():
        yield tuple(
            _process_arg(x, typecode)
            for x, typecode in zip(row, in_types)
        )
