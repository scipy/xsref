# xsref

## Introduction and Installation

This package is used to generate test cases for scalar kernels in the xsf
special function library. Reference values are computed using
[`mpmath`](https://mpmath.org/). Reference tables are stored in
[parquet](https://parquet.apache.org/docs/file-format/) files within the
`xref` repo itself at `xsref/tables/`. Since `xsref` relies on these parquet
files being part of the repo itself, it should be installed using an inplace
build

```bash
pip install -e .
```

from inside the top level directory of the `xsf` repo. Or

```bash
pip install -e .[test]
```

if one wants to install the test dependencies.

## Reference tables

Subdirectories under `xsref/tables` should correspond to conceptually related
collections of test cases. Subdirectories under these subdirectories should
correspond to special functions, and share a name with the corresponding `xsf`
scalar kernel function name (not the SciPy ufunc name). For example
`xsref/tables/scipy_special_tests` is for parquet files for all test cases
that appeared in the `scipy.special` tests prior to the separation of scalar kernels
into the separate [xsf](https://github.com/scipy/xsf) library.
`xsref/tables/scipy_special_tests/cyl_bessel_i` contains parquet files containing
reference test cases for the [modified Bessel function $I_v$](https://dlmf.nist.gov/10.25#E2)
which corresponds to the SciPy ufunc
[`iv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html).

Within a directory like `xsref/tables/scipy_special_tests/cyl_bessel_i`, for
each type overload, there will be a parquet file for inputs, such as `In_d_d-d.parquet`
[^1]. The substring `d_d-d` says that this is for cases for the
signature `double cyl_bessel_i(double, double)`. There will be a corresponding
parquet file containing reference output values `Out_d_d-d.parquet`. For each
tested platform there will be parquet files like `Err_d_d-d_gcc-linux-x86_x6.parquet`
containing current extended relative error [^2] values for `xsf`'s implementation compared
to the reference implementation. Rather than having some fixed tolerance
standard, we track the current relative error at each snapshot of development history,
and test that it is not made worse by some fixed multiple. The idea is to use just
tests to drive continuous improvement, helping us identify cases where the scalar
kernels could be improved.

The hope is that the reference implementations be independent of the `xsf`
implementations, relying on arbitrary precision calculations based on simple
definitions. In this case, agreement between the `xsf` implementation and the
reference implementation should make us reasonably confident that the `xsf`
implementation is accurate. However, disagreement could occur either due to
flaws in the `xsf` implementation, or flaws in the reference implementation.
Such situations must be investigated on a case by case basis.

There are functions and parameter regions where we do not yet have an
arbitrary precision reference implementation. Currently there is a fallback to
the SciPy implementations from just before the scalar kernels were split
into the separate `xsf` library. The `Out` parquet files contain a boolean column,
`"fallback"` which is True for cases where such a fallback was used. 

## Testing

The test suite uses `pytest` and can be run by invoking the `pytest` command
in a shell from within the top level of the `xsref` repo. Currently the test
suite only checks the consistency of the reference tables.

--

[^1]: For complex valued cases, it would be nice to be able to use NumPy typecodes
      such as `In_d_D-D.parquet`, but this will not work on case insensitive file
	  systems such as the default on MacOS. Instead of `D` we use `cd` for complex
	  double and instead of `F` we use `cf` for complex float. The filename in
	  question here would thus be `In_d_cd-cd.parquet`.

[^2]: Extended relative error is a metric we've devised which given any two floating
      point numbers (or complex floating point numbers), including exceptional values like
	  `NaN`s, infinities, and zeros, will return a non-`NaN` result.
