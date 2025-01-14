import functools
import math
import numpy as np
import scipy
import signal
import threading
import typing
import warnings

from mpmath import mp
from packaging import version
from typing import get_origin
from typing import overload


__all__ = ["reference_implementation", "XSRefFallbackWarning"]


if version.parse(scipy.__version__) >= version.parse("1.16"):
    raise RuntimeError(
        f"SciPy {scipy.__version__} is not an independent reference. SciPy"
        " depends on xsf as of version 1.16."
    )


def get_signature_from_type_hints(type_hints):
    input_types = []
    for key, val in type_hints.items():
        if key != "return":
            if np.issubdtype(val, np.number):
                input_types.append(val)
                continue
            else:
                raise ValueError
        if isinstance(val, typing._GenericAlias) or typing.get_origin(val) is tuple:
            output_types = typing.get_args(val)
        else:
            output_types = (val, )

    return (tuple(input_types), output_types)


def get_signatures_for_function(func):
    type_hints = typing.get_type_hints(func)
    if type_hints:
        input_types, output_types = get_signature_from_type_hints(
            typing.get_type_hints(func)
        )
        return {input_types: output_types}
    signatures = {}
    for overload in typing.get_overloads(func):
        input_types, output_types = get_signature_from_type_hints(
            typing.get_type_hints(overload)
        )
        signatures[input_types] = output_types
    return signatures


def get_general_input_signature_from_args(args):
    signature = []
    for arg in args:
        arg_type = type(arg)
        if np.issubdtype(arg_type, np.integer):
            signature.append(np.integer)
        elif np.issubdtype(arg_type, np.floating):
            signature.append(np.floating)
        elif np.issubdtype(arg_type, np.complexfloating):
            signature.append(np.complexfloating)
    return tuple(signature)


def get_output_types(args, general_output_types):
    smallest_type_map = {
        np.integer: np.int8,
        np.floating: np.float16,
        np.complexfloating: np.complex64,
    }
    largest_real_type = np.result_type(*(arg.real for arg in args))
    if largest_real_type.itemsize > 16:
        raise ValueError(
            "xsf reference implementation received unsupported dtype: "
            " long double types are not supported."
        )
    return tuple(
        np.result_type(smallest_type_map[dtype], largest_real_type).type
        for dtype in general_output_types
    )


def process_args(func, *args):
    """Convert finite precision arguments to arbitrary precison."""
    expected_signatures = get_signatures_for_function(func)
    input_signature = get_general_input_signature_from_args(args)
    if input_signature not in expected_signatures:
        raise ValueError
    general_output_types = expected_signatures[input_signature]
    output_types = get_output_types(args, general_output_types)
    new_args = []
    for x, type_ in zip(args, input_signature):
        if type_ is np.floating:
            if x == 0:
                # mpmath doesn't have signed zeros, so if zero, keep this
                # a float to maintain the sign.
                new_args.append(float(x))
                continue
            new_args.append(mp.mpf(float(x)))
        elif type_ is np.integer:
            new_args.append(mp.mpf(float(x)))
        elif type_ is np.complexfloating:
            # Again mpmath doesn't have signed zeros, so if the real or
            # imaginary part are zero, we just pass through a complex
            # float. This should be kept in mind for the cases where
            # this might matter.
            if (x.real == 0 or x.imag) == 0:
                new_args.append(complex(x))
            else:
                new_args.append(mp.mpc(complex(x)))
    return tuple(new_args), output_types


def process_output(args, output_types):
    """Convert arbitrary precision arguments to finite precision."""
    output = []
    for arg, output_type in zip(args, output_types):
        if isinstance(arg, mp.mpc):
            if np.issubdtype(output_type, np.complexfloating):
                output.append(output_type(arg))
            else:
                if abs(arg.imag) == 0:
                    output.append(output_type(arg.real))
                else:
                    # Expected a real result, but got complex. Convention is to return
                    # nan in this case.
                    output.append(output_type("nan"))
        else:
            if np.issubdtype(output_type, np.floating):
                output.append(output_type(arg))
            else:
                # We expect a complex result but got a real output value. If
                # it's a nan, make sure imaginary part is nan too.
                if mp.isnan(arg):
                    output.append(output_type(math.nan, math.nan))
                else:
                    output.append(output_type(arg))
    output = tuple(output)
    return output[0] if len(output) == 1 else output


def get_timeout_handler(self, funcname):
    def timeout_handler(signum, frame):
        raise TimeoutError(
            f"Reference implementation {funcname} timed out after"
            f" {timeout} seconds."

        )
    return timeout_handler


class XSRefFallbackWarning(UserWarning):
    pass


class reference_implementation:
    def __init__(self, *, dps=100, scipy=None, default_timeout=3, nan_invalid=True):
        self.dps = dps
        self.default_timeout = default_timeout
        self.nan_invalid = nan_invalid
        self.scipy_func = scipy

    def _get_timeout_handler(self, funcname, seconds):
        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Reference implementation {funcname} timed out after"
                f" {seconds} seconds."
            )
        return timeout_handler

    def __call__(self, func):
        overloads = typing.get_overloads(func)
        annotations_code = "import numpy\n"
        for overload_def in overloads:
            type_hints = overload_def.__annotations__
            return_type = type_hints['return']
            return_type = (
                f"numpy.{return_type.__name__}"
                if return_type in [np.integer, np.floating, np.complexfloating]
                else return_type
            )
            input_types = (
                (key, f"numpy.{val.__name__}") for key, val in type_hints.items()
                if key != 'return'
            )
            annotations_code += (
                "@overload\n"
                "def wrapper("
                f"{','.join((f'{arg}: {type_}' for arg, type_ in input_types))})"
                f" -> {return_type}: ...\n"
            )
        exec(annotations_code)

        @functools.wraps(func)
        def wrapper(*args, timeout=self.default_timeout):
            if timeout is not None:
                signal.signal(
                    signal.SIGALRM, self._get_timeout_handler(func.__name__, timeout)
                )
                signal.alarm(timeout)
            try:
                with mp.workdps(self.dps):
                    mp_args, output_types = process_args(func, *args)
                    if self.nan_invalid and any([mp.isnan(x) for x in
                                                 mp_args]):
                        result = tuple(mp.nan for _ in range(len(output_types)))
                    else:
                        result = func(*mp_args)
                        if not isinstance(result, tuple):
                            result = (result, )
                    result = process_output(result, output_types)
            except Exception as e:
                if self.scipy_func is not None:
                    result = self.scipy_func(*args)
                    message = (
                        f"Reference implementation {func.__name__} falling"
                        f" back to SciPy\nwith args: {args},\ndue to exception: "
                        f"{type(e).__name__}"
                    )
                    message += f", {e}" if str(e) else ""
                    warnings.warn(message, XSRefFallbackWarning)
                else:
                    raise e
            finally:
                signal.alarm(0)

            return result

        wrapper.__annotations__ =  typing.get_type_hints(func)
        setattr(wrapper, "_xsref_scipy_func", self.scipy_func)
        setattr(wrapper, "_mp", func)
        return wrapper
