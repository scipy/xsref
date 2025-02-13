import numpy as np
import warnings


__all__ = ["extended_absolute_error", "extended_relative_error"]


def _extended_absolute_error_real(actual, desired):
    dtype = type(desired)
    actual = dtype(actual)
    if actual == desired or (np.isnan(actual) and np.isnan(desired)):
        return dtype(0.0)
    if np.isnan(desired) or np.isnan(actual):
        # If expected nan but got non-NaN or expected non-NaN but got NaN
        # we consider this to be an infinite error.
        return dtype("inf")
    if np.isinf(actual):
        # We don't want to penalize early overflow too harshly, so instead
        # compare with the mythical value nextafter(max_float).
        sgn = np.sign(actual)
        mantissa_bits = np.finfo(dtype).nmant
        max_float = np.finfo(dtype).max
        # max_float * 2**-(mantissa_bits + 1) = ulp(max_float)
        ulp = 2**-(mantissa_bits + 1)
        return abs(
            (sgn * max_float - desired) + sgn * ulp
        )
    if np.isinf(desired):
        sgn = np.sign(desired)
        mantissa_bits = np.finfo(dtype).nmant
        max_float = np.finfo(dtype).max
        # max_float * 2**-(mantissa_bits + 1) = ulp(max_float)
        ulp = 2**-(mantissa_bits + 1)
        return abs(
            (sgn * max_float - actual) + sgn * ulp
        )
    return abs(actual - desired)


def _extended_relative_error_real(actual, desired):
    dtype = type(actual)
    abs_error = _extended_absolute_error_real(actual, desired)
    abs_desired = abs(desired)
    if desired == 0.0:
        # If the desired result is 0.0, normalize by smallest subnormal instead
        # of zero. Some answers are still better than others and we want to guard
        abs_desired = np.finfo(dtype).smallest_subnormal
    elif np.isinf(desired):
        abs_desired = np.finfo(dtype).max
    elif np.isnan(desired):
        # extended_relative_error(nan, nan) = 0, otherwise
        # extended_relative_error(x0, x1) with one of x0 or x1 NaN is infinity.
        abs_desired = dtype(1)
    with warnings.catch_warnings(action="ignore"):
        return abs_error / abs_desired


def _extended_absolute_error_complex(actual, desired):
    dtype = type(desired)
    actual = dtype(actual)
    with warnings.catch_warnings(action="ignore"):
        return np.hypot(
            _extended_absolute_error_real(actual.real, desired.real),
            _extended_absolute_error_real(actual.imag, desired.imag)
        )


def _extended_relative_error_complex(actual, desired):
    abs_error = _extended_absolute_error_complex(actual, desired)
    finfo = np.finfo(type(actual.real))

    desired_real = desired.real
    desired_imag = desired.imag

    if desired_real == 0:
        desired_real = np.copysign(finfo.smallest_subnormal, desired.real)
    elif np.isinf(desired_real):
        desired_real = np.copysign(finfo.max, desired.real)
    elif np.isnan(desired_real):
        desired_real = 1.0

    if desired_imag == 0:
        desired_imag = np.copysign(finfo.smallest_subnormal, desired.imag)
    elif np.isinf(desired_imag):
        desired_imag = np.copysign(finfo.max, desired.imag)
    elif np.isnan(desired_imag):
        desired_imag = 1.0

    desired = type(actual)(desired_real, desired_imag)

    with warnings.catch_warnings(action="ignore"):
        try:
            return abs_error / abs(desired)
        except OverflowError as e:
            # Rescale to handle overflow.
            return (abs_error / 2) / abs(desired / 2)


@np.vectorize
def extended_absolute_error(actual, desired):
    if np.issubdtype(type(actual), np.complexfloating):
        return _extended_absolute_error_complex(actual, desired)
    if np.issubdtype(type(actual), np.floating):
        return _extended_absolute_error_real(actual, desired)
    raise ValueError(
        "Unhandled argument type for extended_absolute_error."
        " Arguments must be a subdtype of np.floating or"
        " np.complexfloating."
    )


@np.vectorize
def extended_relative_error(actual, desired):
    if np.issubdtype(type(actual), np.complexfloating):
        return _extended_relative_error_complex(actual, desired)
    if np.issubdtype(type(actual), np.floating):
        return _extended_relative_error_real(actual, desired)
    raise ValueError(
        "Unhandled argument type for extended_relative_error."
        " Arguments must be a subdtype of np.floating or"
        " np.complexfloating."
    )
