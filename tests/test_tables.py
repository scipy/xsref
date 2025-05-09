import numpy as np
import os
import polars as pl
import pprint
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import warnings

from functools import reduce
from numpy.testing import assert_allclose
from pathlib import Path

import xsref

from xsref.float_tools import extended_relative_error
from xsref.tables import _calculate_checksum, get_input_rows, get_output_rows
from xsref._reference._framework import XSRefFallbackWarning


def get_tables_paths():
    root_tables_path = Path(__file__).parents[1] / "tables"
    output = []
    for input_table_path in root_tables_path.glob("**/In_*.parquet"):
        output_table_path = (
            input_table_path.parent / input_table_path.name.replace("In_", "Out_")
        )
        if not os.path.exists(output_table_path):
            output_table_path = None
        tol_pattern = input_table_path.name.replace(
            "In_", "Err_"
        ).removesuffix(".parquet")
        tol_pattern = rf"{tol_pattern}*.parquet"
        tol_table_paths = list(input_table_path.parent.glob(tol_pattern))
        output.append((input_table_path, output_table_path, tol_table_paths))
    return output


def assert_typecode_matches_datatype(typecode, datatype):
    match typecode:
        case "d":
            assert datatype == pa.float64()
        case "f":
            assert datatype == pa.float32()
        case "i":
            assert datatype == pa.int32()
        case "p":
            # intp types are 32bit on 32bit platforms and 64bit on 64
            # bit platforms. This is the default numpy int. We store
            # these as 64 bit and cast to the correct type at test
            # time.
            assert datatype == pa.int64()
        case _:
            raise ValueError(f"Unsupported typecode {typecode}.")


@pytest.mark.parametrize(
    "input_table_path,output_table_path,tol_table_paths",
    get_tables_paths()
)
class TestTableIntegrity:
    def test_checksums_match(
            self, input_table_path, output_table_path, tol_table_paths
    ):
        # Test that the Sha256 checksum for the input table stored in the
        # output table's metadata matches the actual Sha256 checksum of the
        # input table.
        input_table_checksum_expected = _calculate_checksum(input_table_path)
        output_metadata = pq.read_schema(output_table_path).metadata
        input_table_checksum_observed = (
            output_metadata[b"input_checksum"].decode("ascii")
        )
        assert input_table_checksum_observed == input_table_checksum_expected

        output_table_checksum_expected = _calculate_checksum(output_table_path)

        for tol_table_path in tol_table_paths:
            if "_other.parquet" in tol_table_path.name:
                # Default tol tables not generated in the same way, so
                # consistency tested separately.
                continue
            tol_metadata = pq.read_schema(tol_table_path).metadata
            input_table_checksum_observed = (
                tol_metadata[b"input_checksum"].decode("ascii")
            )
            output_table_checksum_observed = (
                tol_metadata[b"output_checksum"].decode("ascii")
            )
            assert input_table_checksum_observed == input_table_checksum_expected
            assert output_table_checksum_observed == output_table_checksum_expected

    def test_default_tol_table(
            self, input_table_path, output_table_path, tol_table_paths
    ):
        tol_tables = []
        other_tol_table = None
        for tol_table_path in tol_table_paths:
            if "_other.parquet" not in tol_table_path.name:
                tol_tables.append(pl.read_parquet(tol_table_path).to_numpy())
            else:
                other_tol_table = pl.read_parquet(tol_table_path).to_numpy()
        max_error = reduce(lambda x, y: np.maximum(x, y), tol_tables)
        max_error = np.maximum(max_error, np.finfo(other_tol_table.dtype).eps)
        # The default table should be looser than all specific tables, but not
        # too loose. Need special handling for infinite tolerances.
        with np.errstate(over="ignore"):
            assert np.all(
                ((other_tol_table > max_error)
                 & (other_tol_table < 16 * max_error))
                | (np.isinf(other_tol_table) & np.isinf(max_error))
                | (np.isinf(other_tol_table) & np.isinf(16 * max_error))
            )

    def test_consistent_type_signatures_metadata(
        self, input_table_path, output_table_path, tol_table_paths
    ):
        if output_table_path is None:
            return
        # Tests that signature in input table metadata matches that in the
        # output table metadata.
        input_metadata = pq.read_schema(input_table_path).metadata
        output_metadata = pq.read_schema(output_table_path).metadata
        assert input_metadata[b"in"] == output_metadata[b"in"]
        assert input_metadata[b"out"] == output_metadata[b"out"]

        for tol_table_path in tol_table_paths:
            tol_metadata = pq.read_schema(tol_table_path).metadata
            assert tol_metadata[b"in"] == output_metadata[b"in"]
            assert tol_metadata[b"out"] == output_metadata[b"out"]

    def test_consistent_type_signatures_metadata_filename(
        self, input_table_path, output_table_path, tol_table_paths
    ):
        # Tests that signature in the input table filename matches that in the
        # input table metadata.
        input_metadata = pq.read_schema(input_table_path).metadata
        input_table_types_from_filename = (
            input_table_path.name.removesuffix(".parquet")
        ).replace("In_", "").replace("cd", "D").replace("cf", "F").replace("_", "")

        intypes_filename, outtypes_filename = input_table_types_from_filename.split("-")
        assert input_metadata[b"in"] == intypes_filename.encode("ascii")
        assert input_metadata[b"out"] == outtypes_filename.encode("ascii")

    def test_consistent_column_types_input(
        self, input_table_path, output_table_path, tol_table_paths
    ):
        # Test input types in parquet columns match input types in metadata.
        input_schema = pq.read_schema(input_table_path)
        in_typecodes = input_schema.metadata[b"in"].decode("ascii")
        in_types = input_schema.types

        i = 0
        for typecode in in_typecodes:
            if typecode == "F":
                assert in_types[i] == pa.float32()
                assert in_types[i+1] == pa.float32()
                i += 2
            elif typecode == "D":
                assert in_types[i] == pa.float64()
                assert in_types[i+1] == pa.float64()
                i += 2
            else:
                assert_typecode_matches_datatype(typecode, in_types[i])
                i += 1

    def test_consistent_column_types_output(
        self, input_table_path, output_table_path, tol_table_paths
    ):
        # Test input types in parquet columns match input types in metadata.
        output_schema = pq.read_schema(output_table_path)
        out_typecodes = output_schema.metadata[b"out"].decode("ascii")
        out_types = output_schema.types

        i = 0
        for typecode in out_typecodes:
            if typecode == "F":
                assert out_types[i] == pa.float32()
                assert out_types[i+1] == pa.float32()
                i += 2
            elif typecode == "D":
                assert out_types[i] == pa.float64()
                assert out_types[i+1] == pa.float64()
                i += 2
            else:
                assert_typecode_matches_datatype(typecode, out_types[i])
                i += 1


    def test_num_rows_match(
            self, input_table_path, output_table_path, tol_table_paths
    ):
        # Check that the inputs table as the same number of rows as
        # the outputs table.
        if output_table_path is None:
            return
        input_table = pq.read_table(input_table_path)
        output_table = pq.read_table(output_table_path)
        assert len(input_table) == len(output_table)

        for tol_table_path in tol_table_paths:
            tol_table = pq.read_table(tol_table_path)
            assert len(tol_table) == len(output_table)

    def test_working_tree_clean(
            self, input_table_path, output_table_path, tol_table_paths
    ):
        output_metadata = pq.read_schema(output_table_path).metadata
        working_tree_state = output_metadata[b"working_tree_state"].decode("ascii")
        assert working_tree_state == "clean"

        for tol_table_path in tol_table_paths:
            tol_metadata = pq.read_schema(tol_table_path).metadata
            working_tree_state = tol_metadata[b"working_tree_state"].decode("ascii")
            assert working_tree_state == "clean"
