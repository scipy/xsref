import inspect
import os
import csv
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import re

from multiprocessing import Lock
from pathlib import Path


__all__ = ["TracedUfunc"]


class TracedUfunc:
    """Wraps a ufunc and traces all arguments it receives along with metadata.

    This was used to determine all existing scipy.special test cases for
    ufuncs. To use, replace the following lines from scipy/special/__init__.py

        from . import _ufuncs
        from ._ufuncs import *

    with

        from xsref.scipy_case_generation import TracedUfunc

        from . import _ufuncs
        for func_name in dir(_ufuncs):
            if func_name.startswith("__") or func_name in ["seterr", "geterr", "errstate"]:
                continue
        func = getattr(_ufuncs, func_name)
        if not callable(func):
            continue
        setattr(
            _ufuncs,
            func_name,
            TracedUfunc(func, outpath=f"~/special_test_cases/{func_name}.csv")
        )
        from ._ufuncs import *

    and then run

        python dev.py test -t scipy.special.tests -m full

    to run all SciPy special tests.

    A csvfile for each tested ufunc will appear in the folder f"~/special_test_cases".
    The first n columns of this csvfile correspond to the n arguments of the function.
    The final three columns contain

        1. A numpy typecode signature of the form "dddD->D", "d->d", "f->f", "pd->d"
           etc. showing the typecodes that were dispatched to for the collection
           of arguments from this row.
        2. The name of the file from which the test was taken.
        3. The name of the test function or method from which the test case
           was taken.

    Here are some example rows from yn.csv for the function ``yn``.

        "1","1","pd->d","special/tests/test_basic.py","test_yn"
        "0","0.1","pd->d","special/tests/test_basic.py","test_y0"
        "1","1","pd->d","special/tests/test_basic.py","test_yn"
        "0","0.1","pd->d","special/tests/test_basic.py","test_y0"
        "1","1","pd->d","special/tests/test_basic.py","test_yn"
    """
    def __init__(self, ufunc, /, *, outpath=None):
        self.__ufunc = ufunc
        self.__outpath = Path(outpath).expanduser()
        self.__lock = Lock()
        self._dtype_map = {
            "float32": "f",
            "float64": "d",
            "float128": "g",
            "complex64": "F",
            "complex128": "D",
            "complex256": "G",
            "int64": "p",
            "int32": "i",
        }

    def __call__(self, *args, **kwargs):
        try:
            expanded_args = np.broadcast_arrays(*args)
            # There is a test that inputs will not broadcast which
            # is asserted to raise a ValueError. Just skip that case.
            # It's not relevant when directly testing scalar kernels.
        except ValueError:
            return self.__ufunc(*args, **kwargs)
        dtypes = tuple(val.dtype for val in expanded_args)
        dtypes = self.__ufunc.resolve_dtypes(dtypes + (None, ) * self.__ufunc.nout)
        dtypes = [self._dtype_map[str(dtype)] for dtype in dtypes]
        sig = "".join(dtypes[:self.__ufunc.nin]) + "->"
        sig += "".join(dtypes[self.__ufunc.nin:])
        expanded_args = [val.flatten() for val in expanded_args]
        rows = (
            row + (sig, ) + self._get_file_metadata() for row in zip(*expanded_args)
        )
        self.__outpath.parent.mkdir(exist_ok=True, parents=True)
        with self.__lock:
            with open(self.__outpath, 'a', newline='') as csvfile:
                csv.writer(csvfile, dialect="unix").writerows(rows)

        return self.__ufunc(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__ufunc, name)

    def _get_file_metadata(self):
        frame = inspect.currentframe()
        pattern1 = re.compile(r"[^/]+/tests/.+\.py")
        pattern2 = re.compile("^test_.*")
        for _ in range(10):
            test_name = frame.f_code.co_name
            test_file = frame.f_globals.get("__file__")
            if test_file is None:
                return test_file, test_name
            test_file = os.path.join(*test_file.split(os.path.sep)[-3:])
            if pattern1.match(test_file) and pattern2.match(test_name):
                return test_file, test_name
            frame = frame.f_back
        return None, None


def traced_cases_to_parquet(funcname, infiles, outpath, filename_prefix):
    """Take a csv produced by TracedUfunc and produce parquet reference table."""
    os.makedirs(outpath, exist_ok=True)
    dtype_map = {
        "f": np.float32,
        "d": np.float64,
        "F": np.complex64,
        "D": np.complex128,
        "p": np.int64,
        "i": np.int32,
    }
    if isinstance(infiles, str):
        infiles = [infiles]

    new_rows = defaultdict(list)
    for infile in infiles:
        with open(infile, 'r', newline='') as csvfile:
            for row in csv.reader(csvfile, dialect="unix"):
                args = row[:-3]
                types = row[-3]
                in_types, out_types = types.split("->")
                if len(args) != len(in_types):
                    continue
                try:
                    args = [
                        str(dtype_map[typecode](arg)) for typecode, arg in zip(in_types, args)
                    ]
                except KeyError:
                    continue
                new_rows[types].append(args)

        for types, rows in new_rows.items():
            in_types, _ = types.split("->")
            dtypes = [dtype_map[typecode] for typecode in in_types]
            df = pl.DataFrame(rows, orient="row").unique()
            df.columns = [f"in{i}" for i in range(len(df.columns))]
            df = df.with_columns(
                *(
                    _parse_column(df[colname], dtype).alias(f"in{i}")
                    for i, (colname, dtype) in enumerate(zip(df.columns, dtypes))
                )
            )
            df = df.to_arrow()
            types = types.replace("->", "-")
            in_types, out_types = types.split("-")
            metadata = {
                b"in": in_types.encode("ascii"),
                b"out": out_types.encode("ascii"),
                b"function": funcname.encode("ascii")
            }
            df = df.replace_schema_metadata(metadata)
            pq.write_table(df,
                os.path.join(outpath, f"{filename_prefix}_{types}.parquet"),
                compression="zstd",
                compression_level=22,
            )
