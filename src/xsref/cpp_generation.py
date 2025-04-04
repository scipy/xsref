"""Generate cpp code for xsf tests"""
import argparse
import pyarrow.parquet as pq
import subprocess

from pathlib import Path

import xsref


func_name_to_header = {
    "airy": "xsf/airy.h",
    "airye": "xsf/airy.h",
    "bdtr": "xsf/stats.h",
    "bdtrc": "xsf/stats.h",
    "bdtri": "xsf/stats.h",
    "bei": "xsf/kelvin.h",
    "beip": "xsf/kelvin.h",
    "ber": "xsf/kelvin.h",
    "berp": "xsf/kelvin.h",
    "besselpoly": "xsf/bessel.h",
    "beta": "xsf/beta.h",
    "betaln": "xsf/beta.h",
    "binom": "xsf/binom.h",
    "cbrt": "xsf/cephes/cbrt.h",
    "cem": "xsf/mathieu.h",
    "cem_cva": "xsf/mathieu.h",
    "chdtr": "xsf/stats.h",
    "chdtrc": "xsf/stats.h",
    "chdtri": "xsf/stats.h",
    "cosdg": "xsf/trig.h",
    "cosm1": "xsf/trig.h",
    "cospi": "xsf/trig.h",
    "cotdg": "xsf/trig.h",
    "cyl_bessel_i": "xsf/bessel.h",
    "cyl_bessel_i0": "xsf/bessel.h",
    "cyl_bessel_i0e": "xsf/bessel.h",
    "cyl_bessel_i1": "xsf/bessel.h",
    "cyl_bessel_i1e": "xsf/bessel.h",
    "cyl_bessel_ie": "xsf/bessel.h",
    "cyl_bessel_j": "xsf/bessel.h",
    "cyl_bessel_j0": "xsf/bessel.h",
    "cyl_bessel_j1": "xsf/bessel.h",
    "cyl_bessel_je": "xsf/bessel.h",
    "cyl_bessel_k": "xsf/bessel.h",
    "cyl_bessel_k0": "xsf/bessel.h",
    "cyl_bessel_k0e": "xsf/bessel.h",
    "cyl_bessel_k1": "xsf/bessel.h",
    "cyl_bessel_k1e": "xsf/bessel.h",
    "cyl_bessel_ke": "xsf/bessel.h",
    "cyl_bessel_y": "xsf/bessel.h",
    "cyl_bessel_y0": "xsf/bessel.h",
    "cyl_bessel_y1": "xsf/bessel.h",
    "cyl_bessel_ye": "xsf/bessel.h",
    "cyl_hankel_1": "xsf/bessel.h",
    "cyl_hankel_1e": "xsf/bessel.h",
    "cyl_hankel_2": "xsf/bessel.h",
    "cyl_hankel_2e": "xsf/bessel.h",
    "dawsn": "xsf/erf.h",
    "digamma": "xsf/digamma.h",
    "ellipe": "xsf/ellip.h",
    "ellipeinc": "xsf/ellip.h",
    "ellipj": "xsf/ellip.h",
    "ellipk": "xsf/ellip.h",
    "ellipkinc": "xsf/ellip.h",
    "ellipkm1": "xsf/ellip.h",
    "erf": "xsf/erf.h",
    "erfc": "xsf/erf.h",
    "erfcinv": "xsf/cephes/erfinv.h",
    "erfcx": "xsf/erf.h",
    "erfi": "xsf/erf.h",
    "exp1": "xsf/expint.h",
    "exp10": "xsf/exp.h",
    "exp2": "xsf/exp.h",
    "expi": "xsf/expint.h",
    "expit": "xsf/log_exp.h",
    "expm1": "xsf/exp.h",
    "expn": "xsf/cephes/expn.h",
    "exprel": "xsf/log_exp.h",
    "fdtr": "xsf/stats.h",
    "fdtrc": "xsf/stats.h",
    "fdtri": "xsf/stats.h",
    "fresnel": "xsf/fresnel.h",
    "gamma": "xsf/gamma.h",
    "gammainc": "xsf/gamma.h",
    "gammaincc": "xsf/gamma.h",
    "gammainccinv": "xsf/gamma.h",
    "gammaincinv": "xsf/gamma.h",
    "gammaln": "xsf/gamma.h",
    "gammasgn": "xsf/gamma.h",
    "gdtr": "xsf/stats.h",
    "gdtrc": "xsf/stats.h",
    "gdtrib": "xsf/cdflib.h",
    "hyp1f1": "xsf/specfun.h",
    "hyp2f1": "xsf/hyp2f1.h",
    "it1i0k0": "xsf/bessel.h",
    "it1j0y0": "xsf/bessel.h",
    "it2i0k0": "xsf/bessel.h",
    "it2j0y0": "xsf/bessel.h",
    "it2struve0": "xsf/struve.h",
    "itairy": "xsf/airy.h",
    "itmodstruve0": "xsf/struve.h",
    "itstruve0": "xsf/struve.h",
    "iv_ratio": "xsf/iv_ratio.h",
    "iv_ratio_c": "xsf/iv_ratio.h",
    "kei": "xsf/kelvin.h",
    "keip": "xsf/kelvin.h",
    "kelvin": "xsf/kelvin.h",
    "ker": "xsf/kelvin.h",
    "kerp": "xsf/kelvin.h",
    "kolmogc": "xsf/stats.h",
    "kolmogci": "xsf/stats.h",
    "kolmogi": "xsf/stats.h",
    "kolmogorov": "xsf/stats.h",
    "kolmogp": "xsf/stats.h",
    "lambertw": "xsf/lambertw.h",
    "lanczos_sum_expg_scaled": "xsf/cephes/lanczos.h",
    "lgam1p": "xsf/cephes/unity.h",
    "log1p": "xsf/log.h",
    "log1pmx": "xsf/log.h",
    "log_expit": "xsf/log_exp.h",
    "log_wright_bessel": "xsf/wright_bessel.h",
    "loggamma": "xsf/gamma.h",
    "logit": "xsf/log_exp.h",
    "mcm1": "xsf/mathieu.h",
    "mcm2": "xsf/mathieu.h",
    "modified_fresnel_minus": "xsf/fresnel.h",
    "modified_fresnel_plus": "xsf/fresnel.h",
    "msm1": "xsf/mathieu.h",
    "msm2": "xsf/mathieu.h",
    "nbdtr": "xsf/stats.h",
    "nbdtrc": "xsf/stats.h",
    "ndtr": "xsf/stats.h",
    "ndtri": "xsf/stats.h",
    "oblate_aswfa": "xsf/sphd_wave.h",
    "oblate_aswfa_nocv": "xsf/sphd_wave.h",
    "oblate_radial1": "xsf/sphd_wave.h",
    "oblate_radial1_nocv": "xsf/sphd_wave.h",
    "oblate_radial2": "xsf/sphd_wave.h",
    "oblate_radial2_nocv": "xsf/sphd_wave.h",
    "oblate_segv": "xsf/sphd_wave.h",
    "owens_t": "xsf/stats.h",
    "pbdv": "xsf/par_cyl.h",
    "pbvv": "xsf/par_cyl.h",
    "pbwa": "xsf/par_cyl.h",
    "pdtr": "xsf/stats.h",
    "pdtrc": "xsf/stats.h",
    "pdtri": "xsf/stats.h",
    "pmv": "xsf/specfun.h",
    "poch": "xsf/cephes/poch.h",
    "prolate_aswfa": "xsf/sphd_wave.h",
    "prolate_aswfa_nocv": "xsf/sphd_wave.h",
    "prolate_radial1": "xsf/sphd_wave.h",
    "prolate_radial1_nocv": "xsf/sphd_wave.h",
    "prolate_radial2": "xsf/sphd_wave.h",
    "prolate_radial2_nocv": "xsf/sphd_wave.h",
    "prolate_segv": "xsf/sphd_wave.h",
    "radian": "xsf/trig.h",
    "rgamma": "xsf/loggamma.h",
    "riemann_zeta": "xsf/zeta.h",
    "round": "xsf/cephes/round.h",
    "scaled_exp1": "xsf/expint.h",
    "sem": "xsf/mathieu.h",
    "sem_cva": "xsf/mathieu.h",
    "shichi": "xsf/sici.h",
    "sici": "xsf/sici.h",
    "sindg": "xsf/trig.h",
    "sinpi": "xsf/trig.h",
    "smirnov": "xsf/stats.h",
    "smirnovc": "xsf/stats.h",
    "smirnovci": "xsf/stats.h",
    "smirnovi": "xsf/stats.h",
    "smirnovp": "xsf/stats.h",
    "spence": "xsf/cephes/spence.h",
    "struve_h": "xsf/struve.h",
    "struve_l": "xsf/struve.h",
    "tandg": "xsf/trig.h",
    "voigt_profile": "xsf/erf.h",
    "wofz": "xsf/erf.h",
    "wright_bessel": "xsf/wright_bessel.h",
    "xlog1py": "xsf/log.h",
    "xlogy": "xsf/log.h",
    "zeta": "xsf/zeta.h",
    "zetac": "xsf/zeta.h"
}


class _generate_test_case:
    def __init__(
        self, func_name, in_types, out_types, case_family, *, arg_names=None
    ):
        self.func_name = func_name
        self.in_types = in_types
        self.out_types = out_types
        self.case_family = case_family
        self.__type_map_in_out = {
            "d": "double",
            "f": "float",
            "D": "std::complex<double>",
            "F": "std::complex<float>",
            "i": "std::int32_t",
            "p": "std::ptrdiff_t",
        }
        self.__type_map_tol = {
            "d": "double",
            "f": "float",
            "D": "double",
            "F": "float",
        }

        if arg_names is None:
            arg_names = [f"in{i}" for i, _ in enumerate(in_types)]
        self.arg_names = arg_names

    def _get_generator_types(self):
        input_ = [self.__type_map_in_out[typecode] for typecode in self.in_types]
        if len(input_) > 1:
            input_ = f"std::tuple<{','.join(input_)}>"
        else:
            input_ = input_[0]

        output = (
            [self.__type_map_in_out[typecode] for typecode in self.out_types] + ['bool']
        )
        output = f"std::tuple<{', '.join(output)}>"

        tol = [self.__type_map_tol[typecode] for typecode in self.out_types]
        if len(tol) > 1:
            tol = f"std::tuple<{', '.join(tol)}>"
        else:
            tol = tol[0]

        return input_, output, tol

    def _get_filenames(self):
        in_types = "_".join(self.in_types).replace("D", "cd").replace("F", "cf")
        out_types = "_".join(self.out_types).replace("D", "cd").replace("F", "cf")
        types = f"{in_types}-{out_types}"
        return (
            f'"In_{types}.parquet"',
            f'"Out_{types}.parquet"',
            f'("Err_{types}_" + get_platform_str() + ".parquet")'
        )

    def _make_generator(self):
        in_file, out_file, tol_file = self._get_filenames()
        input_types, output_types, tol_types = self._get_generator_types()
        result = "auto [input, output, tol] = GENERATE(\n"
        result += "xsf_test_cases<\n"
        result += f"{input_types}, {output_types}, {tol_types}>(\n"
        result += f"tables_path / {in_file},\n"
        result += f"tables_path / {out_file},\n"
        result += f"tables_path / {tol_file}\n"
        result += "));\n"
        return result

    def _make_body(self):
        result = ""
        arg_names = self.arg_names
        if len(arg_names) == 1:
            args = arg_names[0]
            result += f"auto {args} = input;\n"
        else:
            args = ', '.join(arg_names)
            result += f"auto [{args}] = input;\n"
        out_types = self.out_types
        if len(out_types) == 1:
            out_names = ["out"]
            desired_names = ["desired"]
            tol_names = ["tol"]
            error_names = ["error"]
            result += f"auto [desired, fallback] = output;\n"
            result += f"auto out = xsf::{self.func_name}({args});\n"
            result += f"auto error = xsf::extended_relative_error(out, desired);\n"
            result += "tol = adjust_tolerance(tol);\n"
            result += f"CAPTURE({', '.join(arg_names + ['out', 'desired', 'error', 'tol', 'fallback'])});\n"
            result += "REQUIRE(error <= tol);\n"
        else:
            out_names = [f"out{i}" for i, _ in enumerate(out_types)]
            desired_names = [f"desired{i}" for i, _ in enumerate(out_types)]
            desired_unpack = ", ".join(desired_names + ["fallback"])
            tol_names = [f"tol{i}" for i, _ in enumerate(out_types)]
            tol_unpack = ", ".join(tol_names)
            error_names = [f"error{i}" for i, _ in enumerate(out_types)]
            result += f"auto [{desired_unpack}] = output;\n\n"
            for typecode, name in zip(out_types, out_names):
                out_type = self.__type_map_in_out[typecode]
                result += f"{out_type} {name};\n"
            result += "\n"
            # Need references of output variables since these get modified.
            result += f"xsf::{self.func_name}({', '.join(arg_names + out_names)});\n"
            result += f"auto [{tol_unpack}] = tol;\n\n"
            for error_name, out_name, desired_name, tol_name in zip(
                    error_names, out_names, desired_names, tol_names
            ):
                result += f"auto {error_name} = xsf::extended_relative_error({out_name}, {desired_name});\n"
                result += f"{tol_name} = adjust_tolerance({tol_name});\n"
                result += f"CAPTURE({', '.join(arg_names + [out_name, desired_name, error_name, tol_name, 'fallback'])});\n"
                result += f"REQUIRE({error_name} <= {tol_name});\n\n"
        return result

    def __call__(self):
        types = f"{self.in_types}->{self.out_types}"
        func_name = self.func_name
        case_family = self.case_family

        result = f'TEST_CASE("{func_name} {types} {case_family}", "[{func_name}][{types}][{case_family}]")'
        result += " {\n"
        result += "SET_FP_FORMAT()\n"
        result += self._make_generator()
        result += "\n"
        result += self._make_body()
        result += "}\n"
        return result


def generate_cpp_test_case(func_name, in_types, out_types, case_family, *, arg_names=None):
    gen = _generate_test_case(func_name, in_types, out_types, case_family, arg_names=arg_names)
    return gen()


def generate_test_file(func_name, types, case_family, *, arg_names=None):
    result = '#include "../testing_utils.h"\n\n'
    header_name = func_name_to_header[func_name]

    result += f"#include <{header_name}>\n"
    result += "\n"

    result += "namespace fs = std::filesystem;\n"
    result += "\n"

    result += f'fs::path tables_path{{fs::path(XSREF_TABLES_PATH) / "{case_family}" / "{func_name}"}};\n'

    if "xsf/cephes/" in header_name:
        func_name = f"cephes::{func_name}"

    for in_types, out_types in types:
        result += "\n"
        result += generate_cpp_test_case(
            func_name, in_types, out_types, case_family, arg_names=arg_names
        )

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate c++ test cases associated to a collection of parquet"
            " tables from xsref."
        )
    )

    parser.add_argument(
        "inpath_root",
        type=str,
        help="The root directory where parquet files are located."
    )

    parser.add_argument(
        "outpath_root",
        type=str,
        help=(
            "The root directory under which test files will be stored."
            " Typically '~/xsf/tests'."
        )
    )

    parser.add_argument(
        "case_family",
        type=str,
        help="Name for family of test cases being added.",
    )

    parser.add_argument(
        "--force", action="store_true",
        help="If true, recreate files that already exist."
    )

    args = parser.parse_args()

    inpath_root = Path(args.inpath_root)
    outpath_root = Path(args.outpath_root)
    outpath_root.mkdir(exist_ok=True, parents=True)

    func_types = {}
    func_arg_names = {}
    for input_file_path in inpath_root.glob("**/In_*.parquet"):
        metadata = pq.read_schema(input_file_path).metadata
        in_types = metadata[b"in"].decode("ascii")
        out_types = metadata[b"out"].decode("ascii")
        func_name = metadata[b"function"].decode("ascii")

        if func_name not in func_name_to_header:
            continue

        if func_name not in func_types:
            func_types[func_name] = [[in_types, out_types]]
            arg_names = None
            if hasattr(xsref, func_name):
                arg_names = getattr(xsref, func_name)._arg_names
            func_arg_names[func_name] = arg_names

    for func_name, types in func_types.items():
        arg_names = func_arg_names[func_name]
        outpath = outpath_root / f"test_{func_name}.cpp"
        if not outpath.exists() or args.force:
            test_file = generate_test_file(
                func_name, types, args.case_family, arg_names=arg_names
            )
            with open(outpath, "w") as f:
                f.write(test_file)
