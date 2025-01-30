import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path

from xsref.tables import _calculate_checksum


def get_tables_paths():
    root_tables_path = Path(__file__).parents[1] / "tables"
    output = []
    for input_table_path in root_tables_path.glob("**/In_*.parquet"):
        output_table_path = (
            input_table_path.parent / input_table_path.name.replace("In_", "Out_")
        )
        if not os.path.exists(output_table_path):
            output_table_path = None
        output.append((input_table_path, output_table_path))
    return output


def assert_typecode_matches_datatype(typecode, datatype):
    match typecode:
        case "d":
            assert datatype == pa.float64()
        case "f":
            assert datatype == pa.float32()
        case "D":
            assert datatype == pa.struct(
                [("real", pa.float64()), ("imag", pa.float64())]
            )
        case "F":
            assert datatype == pa.struct(
                [("real", pa.float32()), ("imag", pa.float32())]
            )
        case "i":
            assert datatype == pa.int32()
        case "p":
            # intp types are 32bit on 32bit platforms and 64bit on 64
            # bit platforms. This is the default numpy int. We store
            # these as 64 bit and cast to the correct type and test
            # time.
            assert datatype == pa.int64()
        case _:
            raise ValueError(f"Unsupported typecode {typecode}.")


@pytest.mark.parametrize("input_table_path,output_table_path", get_tables_paths())
class TestTableIntegrity:
    def test_checksums_match(self, input_table_path, output_table_path):
        # Test that the Sha256 checksum for the input table stored in the
        # output table's metadata matches the actual Sha256 checksum of the
        # input table.
        if output_table_path is None:
            return
        input_table_checksum_expected = _calculate_checksum(input_table_path)
        output_metadata = pq.read_schema(output_table_path).metadata
        input_table_checksum_observed = (
            output_metadata[b"input_checksum"].decode("ascii")
        )
        assert input_table_checksum_observed == input_table_checksum_expected

    def test_consistent_type_signatures_metadata(
        self, input_table_path, output_table_path
    ):
        # Tests that signature in input table metadata matches that in the
        # output table metadata.
        input_metadata = pq.read_schema(input_table_path).metadata
        output_metadata = pq.read_schema(output_table_path).metadataa
        assert input_metadata[b"in"] == output_metadata[b"in"]
        assert input_metadata[b"out"] == output_metadata[b"out"]

    def test_consistent_type_signatures_filenames(
        self, input_table_path, output_table_path
    ):
        # Tests that signature in input table filename matches that in the
        # output table filename.
        input_table_types_from_filename = (
            input_table_path.name.removesuffix(".parquet").split("_")[1]
        )
        output_table_types_from_filename = (
            output_table_path.name.removesuffix(".parquet").split("_")[1]
        )
        assert input_table_types_from_filename == output_table_types_from_filename

    def test_consistent_type_signatures_metadata_filename(
        self, input_table_path, output_table_path
    ):
        # Tests that signature in the input table filename matches that in the
        # input table metadata.
        input_metadata = pq.read_schema(input_table_path).metadata
        output_metadata = pq.read_schema(output_table_path).metadata
        input_table_types_from_filename = (
            input_table_path.name.removesuffix(".parquet").split("_")[1]
        )
        output_table_types_from_filename = (
            output_table_path.name.removesuffix(".parquet").split("_")[1]
        )
        intypes_filename, outtypes_filename = input_table_types_from_filename.split("-")
        assert input_metadata[b"in"] == intypes_filename.encode("ascii")
        assert input_metadata[b"out"] == outtypes_filename.encode("ascii")

    def test_consistent_column_tables:
        # Test input and output types in parquet columns match input and
        # output types in metadata and filenames.
        input_schema = pq.read_schema(input_table_path)
        output_schema = pq.read_schema(output_table_path)
        in_types = input_schema.metadata[b"in"].decode("ascii")
        out_types = input_schema.metadata[b"out"].decode("ascii")
        for typecode, datatype in zip(in_types, input_schema.types):
            assert_typecode_matches_datatype(typecode, datatype)
        for typecode, datatype in zip(out_types, output_schema.types):
            assert_typecode_matches_datatype(typecode, datatype)
