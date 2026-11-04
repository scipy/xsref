import argparse
import csv
import hashlib
import importlib.resources
import mpmath
import numpy as np
import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import re
import scipy
import subprocess
import warnings


from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from io import StringIO
from multiprocessing import Lock
from multiprocessing import Manager
from pathlib import Path
from tempfile import TemporaryDirectory


import xsref._reference._functions as xsref_funcs

from xsref.float_tools import extended_relative_error
from xsref._reference._framework import XSRefFallbackWarning

_table_revision = ContextVar("_table_revision", default=None)


@contextmanager
def _tables_at_commit(revision):
    """Temporarily use reference tables from a Git revision."""
    if revision is None:
        yield
        return

    repo_root = Path(subprocess.check_output(
        ["git", "-C", str(get_tables_path()), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip())

    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "--verify",
         f"{revision}^{{commit}}"],
        text=True,
    ).strip()

    with TemporaryDirectory() as tmpdir:
        token = _table_revision.set((repo_root, commit, Path(tmpdir)))
        try:
            yield
        finally:
            _table_revision.reset(token)


def _resolve_table_path(path):
    """Resolve a table path against the active Git revision."""
    state = _table_revision.get()

    if state is None:
        if not path.is_file():
            raise FileNotFoundError(f"Table not found: {path}")
        return path

    repo_root, commit, tmpdir = state
    relative_path = path.resolve().relative_to(repo_root)

    historical_path = tmpdir / relative_path

    if not historical_path.is_file():
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show",
             f"{commit}:{relative_path.as_posix()}"],
            capture_output=True,
        )

        if result.returncode:
            raise FileNotFoundError(
                f"Table {relative_path} not found at commit {commit}."
            )

        historical_path.parent.mkdir(parents=True, exist_ok=True)
        historical_path.write_bytes(result.stdout)

    return historical_path


def get_tables_path():
    return importlib.resources.files('xsref').joinpath('tables')


def get_path_to_table(
        function, table_type, *, family="scipy_special_tests", dtype_loop, platform
):
    """Return the path to a reference table.

    Parameters
    ----------
    function : str
        Function name, e.g. 'digamma'.
    table_type : {'in', 'out', 'err'}
        Type of table.
    family : str, optional
        Table family. Default is 'scipy_special_tests'.
    dtype_loop : str
        NumPy-style dtype loop, e.g. 'd->d' or 'D->D'.
    platform : str
        Platform identifier, e.g. 'gcc-linux-x86_64' or 'other'.
        Only used for error tables.

    Returns
    -------
    pathlib.Path
        Path to the requested Parquet table.

    Raises
    ------
    ValueError
        If table_type is invalid.
    FileNotFoundError
        If the requested table does not exist.
    """
    if table_type not in ("in", "out", "err"):
        raise ValueError(f"Unknown table type: {table_type!r}")

    loop = _encode_dtype_loop(dtype_loop)

    name = f"{table_type.capitalize()}_{loop}"
    if table_type == "err":
        name += f"_{platform}"

    path = get_tables_path() / family / function / f"{name}.parquet"
    return _resolve_table_path(path)


def _get_vals_from_path(path):
    """Load logical values from a Parquet table as NumPy arrays."""
    table_type = path.name.split("_", 1)[0].lower()

    if table_type == "in":
        rows = get_input_rows(path)
    elif table_type == "out":
        rows = get_output_rows(path)
    elif table_type == "err":
        rows = pl.read_parquet(path).rows()
    else:
        raise ValueError(f"Unknown table type: {table_type!r}")

    result = tuple(np.asarray(col) for col in zip(*rows))
    if len(result) == 1:
        result = result[0]
    return result


def _process_arg(arg, typecode):
    match typecode:
        case "d":
            return np.float64(arg)
        case "f":
            return np.float32(arg)
        case "i":
            return np.int32(arg)
        case "p":
            return np.int64(arg)
        case "F":
            return np.complex64(arg[0], arg[1])
        case "D":
            return np.complex128(arg[0], arg[1])
        case "_":
            raise ValueError(f"Received unhandled typecode: {typecode}")


def _encode_dtype_loop(dtype_loop):
    """Encode a NumPy dtype loop for use in table filenames."""
    inputs, outputs = dtype_loop.split("->")

    def encode(types):
        return "_".join(
            typecode.replace("F", "cf").replace("D", "cd")
            for typecode in types
        )

    return f"{encode(inputs)}-{encode(outputs)}"


def _parquet_as_text(data):
    """Render Parquet data as CSV with metadata for diffing."""
    metadata = pq.read_schema(BytesIO(data)).metadata or {}

    lines = ["# Metadata\n"]
    for key, value in sorted(metadata.items()):
        key = key.decode("utf-8", errors="backslashreplace")
        value = value.decode("utf-8", errors="backslashreplace")
        lines.append(f"# {key}: {value!r}\n")

    lines.append("# Data (CSV)\n")
    csv = pl.read_parquet(BytesIO(data)).write_csv()
    lines.extend(csv.splitlines(keepends=True))

    return lines


def get_in_out_types(table_path):
    """Get input types of function corresponding to table_path

    Parameters
    ----------
    table_path : str
        Path to a parquet table with rows corresponding to arguments to
        a special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Because parquet
        does not support complex types, complex inputs are stored in two columns
        each of the corresponding base type.

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
        are stored in two columns of the corresponding base type.
        
    Returns
    -------
    List of tuple
        Arguments for reference function from reference table.
    """
    input_types, _ = get_in_out_types(table_path)
    table = pl.read_parquet(table_path)

    results = []
    for row in table.iter_rows():
        i = 0
        processed_row = []
        for typecode in input_types:
            if typecode in ["F", "D"]:
                processed_row.append(_process_arg([row[i], row[i+1]], typecode))
                i += 2
            else:
                processed_row.append(_process_arg(row[i], typecode))
                i += 1
        results.append(tuple(processed_row))
    return results


def get_output_rows(table_path):
    """Return test case reference values from outputs parquet table.

    Parameters
    ----------
    table_path : str
        Path to a parquet table with rows corresponding to outputs of a a
        special function and in format used by xsref. float32, float64, int32
        and int64 inputs have corresponding types in parquet. Complex outputs
        are stored in two columns of the corresponding base type. The final
        column is named ``"fallback"`` and is of type ``bool``. ``"fallback"``
        holds value True if the reference value was taken from SciPy or xsf
        itself, and is thus not an independent reference value.  This is done
        on a temporary basis to guard against regressions if a reference
        implementation has not yet been implemented for a function in a given
        parameter regime.
        
    Returns
    -------
    List of tuple
        Outputs for reference function from reference table.
    """
    _, output_types = get_in_out_types(table_path)
    table = pl.read_parquet(table_path)

    results = []
    for row in table.iter_rows():
        i = 0
        processed_row = []
        for typecode in output_types:
            if typecode in ["F", "D"]:
                processed_row.append(_process_arg([row[i], row[i+1]], typecode))
                i += 2
            else:
                processed_row.append(_process_arg(row[i], typecode))
                i += 1
        results.append(tuple(processed_row))
    return results


def get_vals_from_table(
        function, table_type, *, family="scipy_special_tests",
        dtype_loop, platform, revision=None
):
    """Load reference values and metadata from a Parquet table.

    Parameters
    ----------
    function : str
        Function name, e.g. 'digamma'.
    table_type : {'in', 'out', 'err'}
        Type of table to load.
    family : str, optional
        Table family. Default is 'scipy_special_tests'.
    dtype_loop : str
        NumPy-style dtype loop, e.g. 'd->d' or 'D->D'.
    platform : str
        Platform identifier, e.g. 'gcc-linux-x86_64' or 'other'.
        Only used for error tables.
    revision : str, optional
        Git revision from which to load the table. May be a commit hash,
        branch name, or tag. If None (default), use the working tree.

    Returns
    -------
    dict
        Dictionary containing:

        values : numpy.ndarray or tuple of numpy.ndarray
            Logical table columns, with complex values reconstructed.
            A single array is returned for a single logical column.
        metadata : dict
            Parquet schema metadata, with bytes keys and values.
    """
    with _tables_at_commit(revision):
        path = get_path_to_table(
            function, table_type, family=family,
            dtype_loop=dtype_loop, platform=platform
        )
        values = _get_vals_from_path(path)
        metadata = pq.read_schema(path).metadata or {}
        metadata = {
            key.decode("ascii"): value.decode("ascii")
            for key, value in metadata.items()
        }

    return {"values": values, "metadata": metadata}


def table_to_string(table):
    """Format reference table metadata and values as text for diffing.

    Parameters
    ----------
    table : dict
        Dictionary returned by `get_vals_from_table`, containing
        'values' and 'metadata'.

    Returns
    -------
    str
        Table metadata followed by values formatted as CSV.
        Complex values are represented as strings.
    """
    buffer = StringIO()

    buffer.write("# Metadata\n")
    for key, value in sorted(table["metadata"].items()):
        buffer.write(f"# {key}: {value}\n")

    buffer.write("# Values (CSV)\n")
    values = table["values"]
    columns = values if isinstance(values, tuple) else (values,)

    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([f"value{i}" for i in range(len(columns))])
    writer.writerows(zip(*columns))

    return buffer.getvalue()


from difflib import unified_diff


def get_table_diffs(revision1=None, revision2="main"):
    """Compare reference tables between two Git revisions.

    Parameters
    ----------
    revision1, revision2 : str or None, optional
        Git revisions to compare. May be commit hashes, branch names,
        or tags. None denotes the current working tree. Defaults are
        None (working tree) and 'main', respectively.

    Returns
    -------
    dict of str to str
        Dictionary mapping table paths relative to `get_tables_path()`
        to unified diffs of their metadata and logical values.
        Only tables with differences are included.
    """
    if revision1 is None and revision2 is None:
        return {}

    tables_path = Path(get_tables_path()).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tables_path), *args]
        )

    def paths(output):
        return {
            os.fsdecode(path)
            for path in output.split(b"\0")
            if path
        }

    repo_root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    relative_dir = tables_path.relative_to(repo_root).as_posix()

    def resolve(revision):
        if revision is None:
            return None
        return git(
            "rev-parse", "--verify", f"{revision}^{{commit}}"
        ).decode().strip()

    commit1 = resolve(revision1)
    commit2 = resolve(revision2)

    # Find changed files, including untracked files when comparing
    # against the working tree.
    revisions = (
        [commit1, commit2] if commit1 and commit2
        else [commit1 or commit2]
    )
    changed = paths(git(
        "diff", "--no-renames", "--name-only", "-z",
        *revisions, "--", relative_dir,
    ))

    if commit1 is None or commit2 is None:
        changed |= paths(git(
            "ls-files", "--others", "--exclude-standard",
            "-z", "--", relative_dir,
        ))

    def files_at_revision(commit):
        if commit is None:
            return None
        return paths(git(
            "ls-tree", "-r", "--name-only", "-z",
            commit, "--", relative_dir,
        ))

    files1 = files_at_revision(commit1)
    files2 = files_at_revision(commit2)

    def render(path, relative_path, revision, files):
        if files is None:
            if not path.is_file():
                return None
        elif relative_path not in files:
            return None

        with _tables_at_commit(revision):
            resolved_path = _resolve_table_path(path)
            return table_to_string(_read_table(resolved_path))

    diffs = {}

    for relative_path in sorted(changed):
        if not relative_path.endswith(".parquet"):
            continue

        path = repo_root / relative_path
        table_name = path.relative_to(tables_path).as_posix()

        text1 = render(path, relative_path, commit1, files1)
        text2 = render(path, relative_path, commit2, files2)

        diff = "".join(unified_diff(
            (text1 or "").splitlines(keepends=True),
            (text2 or "").splitlines(keepends=True),
            fromfile=(
                f"{revision1 or 'working tree'}:{table_name}"
                if text1 is not None else "/dev/null"
            ),
            tofile=(
                f"{revision2 or 'working tree'}:{table_name}"
                if text2 is not None else "/dev/null"
            ),
        ))

        if diff:
            diffs[table_name] = diff

    return diffs


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

    row = []
    for val in ref_results:
        if np.issubdtype(val, np.complexfloating):
            row.extend([val.real, val.imag])
        else:
            row.append(val)
    row.append(fallback)

    return row

_shared_lock = Lock()


def _calculate_checksum(filepath):
    with open(filepath, "rb") as f:
        content = f.read()
        checksum = hashlib.sha256(content).hexdigest()
    return checksum


def _get_git_info():
    # Getting a commit hash requires an in-place build.
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).strip()

        status_check = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).parent
        ).decode("utf-8").splitlines()

        pattern = re.compile(r"^ \S{1,2} tables/")

        non_table_changes = [
            line for line in status_check
            if not pattern.search(line)
        ]
        working_tree = b"dirty" if non_table_changes else b"clean"

    except subprocess.CalledProcessError:
        commit_hash = b""
        working_tree = b""
    return commit_hash, working_tree


_np_to_pl_type_map = {
    "f": pl.Float32,
    "d": pl.Float64,
    # complex values (F and D) are stored in two columns of a real floating
    # point type.
    "F": pl.Float32,
    "D": pl.Float64,
    "p": pl.Int64,
    "i": pl.Int32,
}


def _create_table_schema(typecodes, prefix):
    schema = {}
    for i, typecode in enumerate(typecodes):
        if typecode in ["F", "D"]:
            schema[f"{prefix}{i}_real"] = _np_to_pl_type_map[typecode]
            schema[f"{prefix}{i}_imag"] = _np_to_pl_type_map[typecode]
        else:
            schema[f"{prefix}{i}"] = _np_to_pl_type_map[typecode]
    return schema


def compute_output_table(inpath, *, logpath=None, ertol=1e-2, nworkers=1):
    """Compute arrow table of outputs associated to parquet file with inputs

    Parameters
    ----------
    inpath : str
        Path to a parquet file of inputs of the kind accepted by xsref.  It has
        columns for each input argument, of type ``f32``, ``f64``, ``int32``,
        ``int64`` for these respective types, with ``complex64`` and
        ``complex128`` arguments expressed in two columns of type ``f32`` or
        ``f64`` respectively. The metadata contains the input types as an ascii
        encoded string of NumPy dtype codes in the field ``b"in"``, the output
        types as a similar string in the field ``b"out"``, and the associated
        `xsref` reference function in the ascii encoded field ``b"function"``

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

    commit_hash, working_tree = _get_git_info()

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

    schema = _create_table_schema(metadata[b"out"].decode("ascii"), "out")
    schema["fallback"] = pl.Boolean()

    table = pl.DataFrame(results, orient="row", schema=schema)
    table = table.to_arrow()
    table = table.replace_schema_metadata(metadata)

    return table


def compute_initial_err_table(inpath):
    """Use SciPy to calculate initial table of expected errors for xsref

    Parameters
    ----------
    inpath : str
        Path to a parquet file with inputs for test cases

    Returns
    -------
    pyarrow.lib.Table

    Rows correspond to test cases and columns to outputs. Entries contain
    the extended relative error between the expected result from the xsref
    reference and what is computed with SciPy. The idea is that the initial
    tolerances for all test cases will be based on the existing extended
    relative errors between xsref and SciPy.
    """
    inpath = Path(inpath)
    outpath = inpath.parent / inpath.name.replace("In_", "Out_")
    if not os.path.exists(outpath):
        return None
    metadata = pq.read_schema(inpath).metadata
    funcname = metadata[b"function"].decode("ascii")
    xsref_func = getattr(xsref_funcs, funcname)
    scipy_func = xsref_func._scipy_func
    
    input_rows = get_input_rows(inpath)
    reference_output_rows = np.asarray(get_output_rows(outpath)).T
    observed_output = np.asarray(scipy_func(*zip(*input_rows)))

    err = extended_relative_error(observed_output, reference_output_rows)

    tol_table = pa.table(
        {f"err{i}": pa.array(err[i, :]) for i in range(err.shape[0])}
    )

    input_checksum = _calculate_checksum(inpath)
    output_checksum = _calculate_checksum(outpath)

    scipy_config = scipy.__config__.show(mode="dicts")
    cpp_compiler = scipy_config["Compilers"]["c++"]
    architecture = scipy_config["Machine Information"]["host"]["cpu"]
    operating_system = scipy_config["Machine Information"]["host"]["system"]

    metadata[b"input_checksum"] = input_checksum.encode("ascii")
    metadata[b"output_checksum"] = output_checksum.encode("ascii")
    metadata[b"scipy_version"] = scipy.__version__.encode("ascii")
    metadata[b"cpp_compiler"] = cpp_compiler["name"].encode("ascii")
    metadata[b"cpp_compiler_version"] = cpp_compiler["version"].encode("ascii")
    metadata[b"architecture"] = architecture.encode("ascii")
    metadata[b"operating_system"] = operating_system.encode("ascii")
    commit_hash, working_tree = _get_git_info()
    metadata[b"xsref_commit_hash"] = commit_hash
    metadata[b"working_tree_state"] = working_tree

    tol_table = tol_table.replace_schema_metadata(metadata)
    return tol_table


def main(inpath_root, *, logpath_root=None, force=False, ertol=1e-2, nworkers=1):
    inpath_root = Path(inpath_root)
    logpath_root = Path(logpath_root)
    logpath_root.mkdir(exist_ok=True, parents=True)
    for inpath in inpath_root.glob("**/In_*.parquet"):
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


# The script below uses the xsref reference implementations to generate parquet
# files with reference outputs corresponding to the input parquet files
# under (potentially recursively) inpath_root. The logs placed under --logpath_root
# in folders that match the directory structure under inpath_root contain cases
# where the difference between the test cases under SciPy 1.15.1 and the reference
# implementation differ. This is particularly useful for the first
# batch of test cases which are taken from the scipy.special tests, because presumably
# these should all pass (except those which were marked xfail).

# Multiprocessing is used to speed things up, but the way it's used could be improved.
# Right now processes can starve when the number of cores is large, because processes
# are split up one input file at a time, and many of the input files are small.
# This could be improved by batching up cases from all input files at once, something
# which should eventually be pursued, but hasn't been yet because this is good
# enough.

# example use


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate reference output parquet files corresponding to input"
            " parquet files under inpath_root (searches directories recursively"
            " for all with filename matching the pattern \"In_*.parquet\" and"
            " generates the corresponding output files as alongside them as "
            " \"Out_*.parquet\"."
        )
    )
    parser.add_argument(
        "inpath_root",
        type=str,
        help="The root directory where input parquet files are located."
    )
    parser.add_argument(
        "--logpath_root",
        type=str,
        default=None,
        help="The directory where log files will be saved. Defaults to None.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if output already exists and checksums match.",
    )
    parser.add_argument(
        "--ertol",
        type=float,
        default=1e-2,
        help="Error tolerance for cases that end up in the logs (default: 1e-2).",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Number of workers to use for computation (default: 1).",
    )

    args = parser.parse_args()
    main(
        args.inpath_root,
        logpath_root=args.logpath_root,
        force=args.force,
        ertol=args.ertol,
        nworkers=args.nworkers,
    )
