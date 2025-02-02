import csv
import hashlib
import mpmath
import numpy as np
import os
import polars as pl
import pyarrow.parquet as pq
import subprocess
import warnings

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Lock
from multiprocessing import Manager
from pathlib import Path

import xsref._reference._functions as xsref_funcs

from xsref.float_tools import extended_relative_error
from xsref._reference._framework import XSRefFallbackWarning


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
            return np.int64(arg)
        case "_":
            raise ValueError(f"Received unhandled typecode: {typecode}")


def get_in_out_types(table_path):
    """Get input types of function corresponding to table_path

    Parameters
    ----------
    table_path : str
        Path to a parquet table with rows corresponding to arguments to
        a special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Complex inputs
        are stored as structs {"real": x, "imag": y} where x and y have the
        corresponding base type.

    Returns
    -------
    tuple of str
        Strings of NumPy dtype typecodes corresponding to the input types
        and output_types respectively for the given reference table.
    """
    metadata = pq.read_schema(table_path).metadata
    return metadata[b"in"].decode("ascii"), metadata[b"out"].decode("ascii")


def get_input_rows(table_path):
    """Return test cases from inputs parquet table.

    Parameters
    ----------
    table_path : str
        Path to a parquet table with rows corresponding to arguments to
        a special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Complex inputs
        are stored as structs {"real": x, "imag": y} where x and y have the
        corresponding base type.
        
    Returns
    -------
    List of tuple
        Arguments for reference function from reference table.
    """
    input_types, _ = get_in_out_types(table_path)
    table = pl.read_parquet(table_path)

    results = []
    for row in table.iter_rows():
        results.append(
            tuple(
                _process_arg(x, typecode)
                for x, typecode in zip(row, input_types)
            )
        )
    return results


def get_output_rows(table_path):
    """Return test case reference values from outputs parquet table.

    Parameters
    ----------
    table_path : str
        Path to a parquet table with rows corresponding to outputs of a a
        special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Complex outputs
        are stored as structs {"real": x, "imag": y} where x and y have the
        corresponding base type. The final column is named ``"fallback"`` and
        is of type ``bool``. ``"fallback"`` holds value True if the reference
        value was taken from SciPy or xsf itself, and is thus not an
        independent reference value.  This is done on a temporary basis to
        guard against regressions if a reference implementation has not yet
        been implemented for a function in a given parameter regime.
        
    Returns
    -------
    List of tuple
        Outputs for reference function from reference table.
    """
    _, output_types = get_in_out_types(table_path)
    table = pl.read_parquet(table_path)

    results = []
    for row in table.iter_rows():
        # Exclude final "fallback" column.
        row = row[:-1]
        results.append(
            tuple(
                _process_arg(x, typecode)
                for x, typecode in zip(row, output_types)
            )
        )
    return results


def init(shared_lock):
    global _shared_lock


def _evaluate(func, logpath, ertol, lock, args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", XSRefFallbackWarning)
        ref_results = func(*args)
        fallback = any([x.category is XSRefFallbackWarning for x in w])

    scipy_results = func._scipy_func(*args)
    if not isinstance(scipy_results, tuple):
        scipy_results = (scipy_results, )
    if not isinstance(ref_results, tuple):
        ref_results = (ref_results, )

    if not len(scipy_results) == len(ref_results):
        raise ValueError(
            f"Reference function {func} returned a different"
            " number of outputs from corresponding SciPy function"
            f" {func._scipy_func} for args {args}."
        )


    if (
            tuple((type(x)) for x in scipy_results)
            != tuple(type(x) for x in ref_results)
    ):
        raise ValueError(
            f"Reference function {func} returned outputs of different"
            " type from corresponding SciPy function"
            f" {func._scipy_func} for args {args}."
        )

    errors = []
    for scipy_result, ref_result in zip(scipy_results, ref_results):
        error = extended_relative_error(scipy_result, ref_result)
        if logpath is not None and error >= ertol:
            with lock:
                if not os.path.exists(logpath):
                    with open(logpath, 'w', newline='') as csvfile:
                        csv.writer(csvfile, dialect="unix").writerow(
                            [f"in{i}" for i in range(len(args))]
                            + [f"ref_out{i}" for i in range(len(ref_results))]
                            + [f"scipy_out{i}" for i in range(len(scipy_results))]
                        )
                with open(logpath, 'a', newline='') as csvfile:
                    csv.writer(csvfile, dialect="unix").writerow(
                        args + ref_results + scipy_results
                    )
    row = [
        val if not np.issubdtype(val, np.complexfloating)
        else {"real": val.real, "imag": val.imag}
        for val in ref_results
    ] + [fallback]
    return row

_shared_lock = Lock()


def _calculate_checksum(filepath):
    with open(filepath, "rb") as f:
        content = f.read()
        checksum = hashlib.sha256(content).hexdigest()
    return checksum


def numpy_typecode_to_polars_type(typecode):
    mapping = {
        "d": pl.Float64,
        "f": pl.Float32,
        "p": pl.Int64,
        "i": pl.Int64,
        "D": pl.Struct({"real": pl.Float64, "imag": pl.Float64}),
        "F": pl.Struct({"real": pl.Float32, "imag": pl.Float32}),
    }
    data_type = mapping.get(typecode)
    if data_type is None:
        raise ValueError(f"Received unsupported typecode, {typecode}")
    return data_type


def compute_output_table(inpath, *, logpath=None, ertol=1e-2, nworkers=1):
    """Compute arrow table of outputs associated to parquet file with inputs

    Parameters
    ----------
    inpath : str
        Path to a parquet file of inputs of the kind accepted by xsref.  It has
        columns for each input argument, of type ``f32``, ``f64``, ``int32``,
        ``int64`` for these respective types, with ``complex64`` and
        ``complex128`` arguments expressed as ``struct{"real": f32, "imag":
        f32}`` and ``struct{"real": f64, "imag": f64}`` respectively. The
        metadata contains the input types as an ascii encoded string of NumPy
        dtype codes in the field ``b"in"``, the output types as a similar
        string in the field ``b"out"``, and the associated `xsref` reference
        function in the ascii encoded field ``b"function"``

    logpath : Optional[str]
        Path to where to store log file for this run. If None, no log is
        stored.  The log is a csv file of rows where the associated SciPy
        function, returns a value which differs from the arbitrary precision
        reference function by an extended relative error greater than
        `ertol`. Each row contains input arguments together with corresponding
        outputs for first the reference implementation and then the SciPy
        implementation. Default: ``None``

    ertol : Optional[float]
        Extended relative error cutoff for adding a row to the log. Extended
        relative error is equal to relative error for non-exceptional values
        (finite and non-zero), but is modified to still produce an informative
        value when a comparison contains one or more exceptional values.
        Default: ``1e-2``

    nworkers : Optional[int]
        Controls the max number of workers used by `ProcessPoolExecutor`.
        Default: ``1``.

    Returns
    -------
    pyarrow.lib.Table
        An arrow table of outputs associated to the corresponding inputs, plus
        a Boolean column ``"fallback"`` which takes value ``True`` if and only
        if the reference computation fell back to a double precision
        calculation in SciPy. The metadata for the input table is the same as
        for the output table, except that there are new entries:

        input_checksum
            containing a sha256 checksum of the input parquet table.
        mpmath_version
            mpmath.__version__ at time of running
        xsref_commit_hash
            If using an in-place build, the current git commit hash for xsref.
        working_tree_state
            One of b"dirty" or b"clean".

       These additional metadata items can be used to help verify the integrity
       of reference tables.
    """
    metadata = pq.read_schema(inpath).metadata
    checksum = _calculate_checksum(inpath)
    metadata[b"input_checksum"] = checksum.encode("ascii")
    metadata[b"mpmath_version"] = mpmath.__version__.encode("ascii")
    # Getting a commit hash requires an in-place build.
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).strip()
        diff_index = subprocess.check_output(
            ["git", "diff-index", "HEAD"], cwd=Path(__file__).parent
        )
        working_tree = b"dirty" if diff_index else b"clean"
    except subprocess.CalledProcessError:
        commit_hash = b""
        working_tree = b""
    metadata[b"xsref_commit_hash"] = commit_hash
    metadata[b"working_tree_state"] = working_tree

    funcname = metadata[b"function"].decode("ascii")
    func = getattr(xsref_funcs, funcname)

    manager = Manager()
    lock = manager.Lock()

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        results = executor.map(
            partial(_evaluate, func, logpath, ertol, lock), get_input_rows(inpath)
        )
        results = list(results)
        if not results:
            return None

    schema = {
        f"out{i}": numpy_typecode_to_polars_type(typecode)
        for i, typecode in enumerate(metadata[b"out"].decode("ascii"))
    }
    schema["fallback"] = pl.Boolean

    table = pl.DataFrame(results, orient="row", schema=schema).to_arrow()
    table = table.replace_schema_metadata(metadata)
    return table


def main(path, *, logpath_root=None, force=False, ertol=1e-2, nworkers=1):
    path = Path(path)
    logpath_root = Path(logpath_root)
    logpath_root.mkdir(exist_ok=True, parents=True)
    for inpath in path.glob("**/In_*.parquet"):
        outpath = inpath.parent / inpath.name.replace("In_", "Out_")
        input_checksum = _calculate_checksum(inpath)
        if os.path.exists(outpath):
            output_metadata = pq.read_schema(outpath).metadata
            if (
                    input_checksum == output_metadata[b"input_checksum"].decode("ascii")
                    and not force
            ):
                continue
        logpath = logpath_root / inpath.relative_to(
            inpath.parents[2]
        ).with_suffix(".log")
        logpath.parent.mkdir(exist_ok=True, parents=True)
        table = compute_output_table(
            inpath, logpath=logpath, ertol=ertol, nworkers=nworkers,
        )
        pq.write_table(table, outpath, compression="zstd", compression_level=22)
