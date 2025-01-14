import math
import numpy as np
import scipy
import scipy.special as special
import sys

from mpmath import mp  # type: ignore
from mpmath.calculus.optimization import Bisection, Secant
from packaging import version
from typing import overload, Tuple

from ._framework import reference_implementation


if version.parse(scipy.__version__) >= version.parse("1.16"):
    raise RuntimeError(
        f"SciPy {scipy.__version__} is not an independent reference. SciPy"
        " depends on xsf as of version 1.16."
        )


Integer = np.integer
Real = np.floating
Complex = np.complexfloating
 


def is_complex(x):
    """Check if x is a relevant complex type

    Note: The only types which can appear as inputs to a reference
    implementation after passing through the argument processing added by the
    decorator are `float`, `mp.mpf`, `complex`, and `mp.mpc` (non mp types can
    appear in order to preserve the sign of zero).
    """
    return isinstance(x, (mp.mpc, complex))


def to_fp(x):
    """Cast mp to finite precision, otherwise idempotent."""
    if isinstance(x, mp.mpf):
        return float(x)
    if isinstance(x, mp.mpc):
        return complex(x)
    return x


def to_mp(x):
    """Cast finite precision to mp, otherwise idempotent."""
    if np.issubdtype(x, Real):
        return mp.mpf(x)
    if np.issubdtype(x, Integer):
        return int(x)
    if np.issubdtype(x, Complex):
        return mp.mpc(x)
    return x


@overload
def airy(x: Real) -> Tuple[Real, Real, Real, Real]: ...
@overload
def airy(x: Complex) -> Tuple[Complex, Complex, Complex, Complex]: ...


@reference_implementation()
def airy(z):
    """Airy functions and their derivatives.

    Notes
    -----
    Airy functions are entire
    """
    ai = mp.airyai(z)
    aip = mp.airyai(z, derivative=1)
    bi = mp.airybi(z)
    bip = mp.airybi(z, derivative=1)
    return ai, aip, bi, bip


@overload
def airye(x: Real) -> Tuple[Real, Real, Real, Real]: ...
@overload
def airye(x: Complex) -> Tuple[Complex, Complex, Complex, Complex]: ...


@reference_implementation()
def airye(z):
    """Exponentially scaled Airy functions and their derivatives.

    Notes
    -----
    Scaled Airy functions are entire
    """
    eai = mp.airyai(z) * mp.exp(mp.mpf("2.0") / mp.mpf("3.0") * z * mp.sqrt(z))
    eaip = mp.airyai(z, derivative=1) * mp.exp(
        mp.mpf("2.0") / mp.mpf("3.0") * z * mp.sqrt(z)
    )
    ebi = mp.airybi(z) * mp.exp(
        -abs(mp.mpf("2.0") / mp.mpf("3.0") * (z * mp.sqrt(z)).real)
    )
    ebip = mp.airybi(z, derivative=1) * mp.exp(
        -abs(mp.mpf("2.0") / mp.mpf("3.0") * (z * mp.sqrt(z)).real)
    )
    return eai, eaip, ebi, ebip


@overload
def bdtr(k: Real, n: Integer, p: Real) -> Real: ...
@overload
def bdtr(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation()
def bdtr(k, n, p):
    """Binomial distribution cumulative distribution function."""
    k, n = mp.floor(k), mp.floor(n)
    if p > 1 or p < 0 or k < 0 or n < 0 or k > n or mp.isinf(n):
        return mp.nan
    if k == n:
        return mp.one
    # set the precision high enough that mp.one - p != 1
    precision = int(mp.ceil(-mp.log(abs(p), b=2))) + 53 if 0 < abs(p) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = betainc._mp(n - k, k + 1, 1 - p)
    return result


@overload
def bdtrc(k: Real, n: Integer, p: Real) -> Real: ...
@overload
def bdtrc(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation()
def bdtrc(k, n, p):
    """Binomial distribution survival function."""
    if p > 1 or p < 0 or k < 0 or n < 0 or k > n or mp.isinf(n):
        return mp.nan
    k, n = mp.floor(k), mp.floor(n)
    if k == n:
        return mp.zero
    return betainc._mp(k + 1, n - k, p)


@overload
def bdtri(k: Real, n: Integer, y: Real) -> Real: ...
@overload
def bdtri(k: Real, n: Real, y: Real) -> Real: ...


@reference_implementation()
def bdtri(k, n, y):
    if y > 1 or y < 0 or k < 0 or n < 0 or k > n or mp.isinf(n):
        return mp.nan
    """Inverse function to `bdtr` with respect to `p`."""
    k, n = mp.floor(k), mp.floor(n)
    def f(p):
        return bdtr._mp(k, n, p) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def bei(x: Real) -> Real:
    """Kelvin function bei."""
    return mp.bei(0, x)


@reference_implementation()
def beip(x: Real) -> Real:
    """Derivative of the Kelvin function bei."""
    return to_mp(special.beip(to_fp(x)))


@reference_implementation()
def ber(x: Real) -> Real:
    """Kelvin function ber."""
    return mp.ber(0, x)


@reference_implementation()
def berp(x: Real) -> Real:
    """Derivative of the Kelvin function ber."""
    return to_mp(special.berp(to_fp(x)))


@reference_implementation()
def besselpoly(a: Real, lmb: Real, nu: Real) -> Real:
    """Weighted integral of the Bessel function of the first kind."""
    def integrand(x):
        return x**lmb * cyl_bessel_j._mp(nu, 2 * a * x)

    return mp.quad(integrand, [0, 1])


@reference_implementation()
def beta(a: Real, b: Real) -> Real:
    """Beta function."""
    return mp.beta(a, b)


@reference_implementation()
def betaln(a: Real, b: Real) -> Real:
    """Natural logarithm of the absolute value of the Beta function."""
    return mp.log(abs(mp.beta(a, b)))


@reference_implementation()
def betainc(a: Real, b: Real, x: Real) -> Real:
    """Regularized incomplete Beta function."""
    return mp.betainc(a, b, 0, x, regularized=True)


@reference_implementation()
def betaincc(a: Real, b: Real, x: Real) -> Real:
    """Complement of the regularized incomplete Beta function."""
    return mp.betainc(a, b, x, 1.0, regularized=True)


@reference_implementation()
def betaincinv(a: Real, b: Real, y: Real) -> Real:
    """Inverse of the regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, 0, x, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def betainccinv(a: Real, b: Real, y: Real) -> Real:
    """Inverse of the complemented regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, x, 1.0, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def binom(n: Real, k: Real) -> Real:
    """Binomial coefficient considered as a function of two real variables."""
    return mp.binomial(n, k)


@reference_implementation()
def cbrt(x: Real) -> Real:
    """Cube root of x."""
    return mp.cbrt(x)


@reference_implementation(uses_mp=False)
def cem(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even Mathieu function and its derivative."""
    return special.mathieu_cem(m, q, x)


@reference_implementation(uses_mp=False)
def cem_cva(m: Real, q: Real) -> Real:
    """Characteristic value of even Mathieu functions."""
    return special.mathieu_a(m, q)


@reference_implementation()
def chdtr(v: Real, x: Real) -> Real:
    """Chi square cumulative distribution function."""
    if x < 0 or v < 0:
        return mp.nan
    return gammainc._mp(v / 2, x / 2)

@reference_implementation()
def chdtrc(v: Real, x: Real) -> Real:
    """Chi square survival function."""
    if x < 0 or v < 0:
        return mp.nan
    return gammaincc._mp(v / 2, x / 2)


@reference_implementation(uses_mp=False)
def chdtri(v: Real, p: Real) -> Real:
    """Inverse to `chdtrc` with respect to `x`."""
    # TODO. Figure out why chdtri inverts chdtrc and not chdtr
    return special.chdtri(v, p)


@reference_implementation(uses_mp=False)
def cosdg(x: Real) -> Real:
    """Cosine of the angle x given in degrees."""
    return special.cosdg(x)


@reference_implementation()
def cosm1(x: Real) -> Real:
    """cos(x) - 1 for use when x is near zero."""
    # set the precision high enough to avoid catastrophic cancellation
    # cos(x) - 1 = x^2/2 + O(x^4) for x near 0
    if not mp.isfinite(x):
        return mp.nan
    precision = (
        min(int(mp.ceil(-2*mp.log(abs(x), b=2))), 1024) + 53 if 0 < abs(x) < 1 else 0
    )
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result =  mp.cos(x) - mp.one
    return result


@overload
def cospi(x: Real) -> Real: ...
@overload
def cospi(x: Complex) -> Complex: ...


@reference_implementation()
def cospi(x):
    """Cosine of pi*x."""
    # This already does the right thing regarding sign of zero for the case
    # x - 0.5 an integer.
    return mp.cospi(x)


@reference_implementation(uses_mp=False)
def cotdg(x: Real) -> Real:
    """Cotangent of the angle x given in degrees."""
    return special.cotdg(x)


@overload
def cyl_bessel_i(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_i(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_i(v, z):
    """Modified Bessel function of the first kind.


    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``
    except for integer `v`.
    """
    if v == mp.floor(v) and v < 0:
        # Use symmetry iv(n, z) == iv(-n, z) for integers n.
        # https://dlmf.nist.gov/10.27#E1
        # mpmath has an easier time with positive v.
        v = -v
    if z.imag == 0 and z.real < 0:
        if v != mp.floor(v):
            if not is_complex(z):
                # We will get a complex value on the branch cut, so if received real
                # input and expecting real output, just return NaN.
                return mp.nan
            # On branch cut, choose branch based on sign of zero
            z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)

    if min(abs(v), abs(z)) > 1e3:
        # mpmath can hang indefinitely for some large values of v and z.
        return to_mp(special.iv(to_fp(v), to_fp(z)))

    try:
        result = mp.besseli(v, z)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.iv(to_fp(v), to_fp(z)))

    if z.imag == 0 and z.real < 0 and v == mp.floor(v):
        # No discontinuity for integer v and should return a real value.
        # Numerical error can cause a small imaginary part here, so just return
        # the real part.
        result = result.real
    return result


@reference_implementation()
def cyl_bessel_i0(z: Real) -> Real:
    """Modified Bessel function of order 0."""
    return cyl_bessel_i._mp(0, z)


@reference_implementation()
def cyl_bessel_i0e(z: Real) -> Real:
    """Exponentially scaled modified Bessel function of order 0."""
    return mp.exp(-abs(z.real)) * cyl_bessel_i0._mp(z)


@reference_implementation()
def cyl_bessel_i1(z: Real)-> Real:
    """Modified Bessel function of order 1."""
    return cyl_bessel_i._mp(1, z)


@reference_implementation()
def cyl_bessel_i1e(z: Real) -> Real:
    """Exponentially scaled modified Bessel function of order 1."""
    return mp.exp(-abs(z.real)) * cyl_bessel_i1._mp(z)


@overload
def cyl_bessel_ie(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ie(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_ie(v, z):
    """Exponentially scaled modified Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``
    except for integer `v`.
    """
    return mp.exp(-abs(z.real)) * cyl_bessel_i._mp(v, z)


@overload
def cyl_bessel_j(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_j(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_j(v, z):
    """Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``
    except for integer `v`.
    """
    if v < 0 and v == mp.floor(v):
        v = -v
        prefactor = (-1)**v
    else:
        prefactor = 1.0

    if z.imag == 0 and z.real < 0 and v != mp.floor(v):
        if v != mp.floor(v):
            if not is_complex(z):
                # We will get a complex value on the branch cut, so if received real
                # input and expecting real output, just return NaN.
                return mp.nan
            # On branch cut, choose branch based on sign of zero
            z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)

    if min(abs(v), abs(z)) > 1e3:
        # mpmath can hang indefinitely for some large values of v and z.
        return prefactor * to_mp(special.jv(to_fp(v), to_fp(z)))
    try:
        result = prefactor * mp.besselj(v, z)
    except (mp.NoConvergence, ValueError):
        return prefactor * to_mp(special.jv(to_fp(v), to_fp(z)))

    if z.imag == 0 and z.real < 0 and v == mp.floor(v):
        # No discontinuity for integer v and should return a real value.
        # Numerical error can cause a small imaginary part here, so just return
        # the real part.
        result = result.real
    return result


@reference_implementation()
def cyl_bessel_j0(x: Real) -> Real:
    """Bessel function of the first kind of order 0."""
    return mp.j0(x)


@reference_implementation()
def cyl_bessel_j1(x: Real) -> Real:
    """Bessel function of the first kind of order 1."""
    return mp.j1(x)


@overload
def cyl_bessel_je(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_je(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_je(v, z):
    """Exponentially scaled Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``
    except for integer `v`.
    """
    return mp.exp(-abs(z.imag)) * cyl_bessel_j._mp(v, z)


@overload
def cyl_bessel_k(v: Integer, z: Real) -> Real: ...
@overload
def cyl_bessel_k(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_k(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_k(v, z):
    """Modified Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``
    except for integer `v`.
    """
    if v < 0:
        v = -v
    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)
    try:
        return mp.besselk(v, z)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.kv(to_fp(v), to_fp(z)))


@reference_implementation()
def cyl_bessel_k0(x: Real) -> Real:
    """Modified Bessel function of the second kinf of order 0."""
    return cyl_bessel_k._mp(0, x)


@reference_implementation()
def cyl_bessel_k0e(x: Real) -> Real:
    """Exponentially scaled modified Bessel function K of order 0."""
    return mp.exp(x) * cyl_bessel_k0._mp(x)


@reference_implementation()
def cyl_bessel_k1(x: Real) -> Real:
    """Modified Bessel function of the second kind of order 0."""
    return cyl_bessel_k._mp(1, x)


@reference_implementation()
def cyl_bessel_k1e(x: Real) -> Real:
    """Exponentially scaled modified Bessel function K of order 1."""
    return mp.exp(x) * cyl_bessel_k1._mp(x)


@overload
def cyl_bessel_ke(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ke(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_ke(v, z):
    """Exponentially scaled modified Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    return mp.exp(z) * cyl_bessel_k._mp(v, z)


@overload
def cyl_bessel_y(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_y(v: Real, z: Complex) -> Complex: ...


@reference_implementation(timeout=10)
def cyl_bessel_y(v, z):
    """Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if v < 0 and v == mp.floor(v):
        # https://dlmf.nist.gov/10.4#E1
        v = -v
        prefactor = (-1)**v
    else:
        prefactor = 1.0

    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)

    if min(abs(v), abs(z)) > 1e3:
        # mpmath can hang indefinitely for some large values of v and z.
        return prefactor * to_mp(special.yv(to_fp(v), to_fp(z)))
    try:
        return prefactor * mp.bessely(v, z)
    except (mp.NoConvergence, ValueError, ZeroDivisionError):
        return prefactor * to_mp(special.yv(to_fp(v), to_fp(z)))


@reference_implementation()
def cyl_bessel_y0(z: Real) -> Real:
    """Bessel function of the second kind of order 0."""
    return cyl_bessel_y._mp(0, z)


@reference_implementation()
def cyl_bessel_y1(z: Real) -> Real:
    """Bessel function of the second kind of order 1."""
    return cyl_bessel_y._mp(1, z)


@overload
def cyl_bessel_ye(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ye(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_bessel_ye(v, z):
    """Exponentially scaled Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    return cyl_bessel_y._mp(v, z) * mp.exp(-abs(z.imag))


@overload
def cyl_hankel_1(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_1(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_hankel_1(v, z):
    """Hankel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)
    try:
        return mp.hankel1(v, z)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.hankel1(to_fp(v), to_fp(z)))


@overload
def cyl_hankel_1e(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_1e(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_hankel_1e(v, z):
    """Exponentially scaled Hankel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    return cyl_hankel_1._mp(v, z) * mp.exp(z * -1j)


@overload
def cyl_hankel_2(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_2(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_hankel_2(v, z):
    """Hankel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)
    try:
        return mp.hankel2(v, z)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.hankel2(to_fp(v), to_fp(z)))


@overload
def cyl_hankel_2e(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_2e(v: Real, z: Complex) -> Complex: ...


@reference_implementation()
def cyl_hankel_2e(v, z):
    """Exponentially scaled Hankel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    return cyl_hankel_2._mp(v, z) * mp.exp(z * 1j)


@overload
def dawsn(x: Real) -> Real: ...
@overload
def dawsn(x: Complex) -> Complex: ...


@reference_implementation()
def dawsn(x):
    """Dawson's integral

    dawsn is an entire function
    """
    def integrand(t):
        return mp.exp(t**2)

    return mp.exp(-x**2) * mp.quad(integrand, [0, x])


@overload
def digamma(x: Real) -> Real: ...
@overload
def digamma(x: Complex) -> Complex: ...


@reference_implementation()
def digamma(x):
    """The digamma function.

    Notes
    -----
    Poles at nonpositive integers.
    """
    if x == 0.0:
        if isinstance(x, float):
            return -math.copysign(mp.inf, x)
        return mp.nan
    if x.real < 0 and x.imag == 0 and x.real == mp.floor(x.real):
        return mp.nan
    return mp.digamma(x)


@reference_implementation()
def ellipe(m: Real) -> Real:
    """Complete elliptic integral of the second kind."""
    return mp.ellipe(m)


@reference_implementation()
def ellipeinc(phi: Real, m: Real) -> Real:
    """Incomplete elliptic integral of the second kind."""
    if not mp.isfinite(phi) or mp.isnan(m) or m == mp.inf:
        # mpmath doesn't handle these cases, use self reference for now.
        return to_mp(special.ellipkinc(to_fp(phi), to_fp(m)))
    return mp.ellipe(phi, m)


@reference_implementation()
def ellipj(u: Real, m: Real) -> Tuple[Real, Real, Real, Real]:
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return sn, cn, dn, phi


@reference_implementation()
def ellipk(m: Real) -> Real:
    """Complete elliptic integral of the first kind."""
    return mp.ellipk(m)


@reference_implementation()
def ellipkinc(phi: Real, m: Real) -> Real:
    """Incomplete elliptic integral of the first kind."""
    if not mp.isfinite(phi) or mp.isnan(m) or m == mp.inf:
        # mpmath doesn't handle these cases, use self reference for now.
        return to_mp(special.ellipkinc(to_fp(phi), to_fp(m)))
    return mp.ellipf(phi, m)


@reference_implementation()
def ellipkm1(p: Real) -> Real:
    """Complete elliptic integral of the first kind around m = 1."""
    # set the precision high enough to resolve 1 - p
    precision = int(mp.ceil(-mp.log(abs(p), b=2))) + 53 if 0 < abs(p) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = ellipk._mp(1 - p)
    return result


@overload
def erf(x: Real) -> Real: ...
@overload
def erf(x: Complex) -> Complex: ...


@reference_implementation()
def erf(x):
    """Error function.

    erf is an entire function
    """
    return mp.erf(x)


@overload
def erfi(x: Real) -> Real: ...
@overload
def erfi(x: Complex) -> Complex: ...


@reference_implementation()
def erfi(x):
    """Imaginary error function.

    erfi is an entire function
    """
    return -mp.j * mp.erf(mp.j * x)


@overload
def erfc(x: Real) -> Real: ...
@overload
def erfc(x: Complex) -> Complex: ...


@reference_implementation()
def erfc(x):
    """Complementary error function 1 - erf(x).

    Notes
    -----
    erfc is an entire function
    """
    return mp.erfc(x)


@overload
def erfcx(x: Real) -> Real: ...
@overload
def erfcx(x: Complex) -> Complex: ...


@reference_implementation()
def erfcx(x):
    """Scaled complementary error function exp(x**2) * erfc(x)

    Notes
    -----
    erfcx is an entire function
    """
    return mp.exp(x**2) * mp.erfc(x)


@reference_implementation()
def erfcinv(x: Real) -> Real:
    """Inverse of the complementary error function."""
    if not 0 <= x <= 2:
        return mp.nan
    if x == 0:
        return mp.inf
    # set the precision high enough to resolve 1 - x.
    precision = int(mp.ceil(-mp.log(abs(x), b=2))) + 53 if 0 < abs(x) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = mp.erfinv(mp.one - x)
    return result


@overload
def exp1(x: Real) -> Real: ...
@overload
def exp1(x: Complex) -> Complex: ...


@reference_implementation()
def exp1(x):
    """Exponential integral E1.

    Notes
    -----
    Logarithmic singularity at x = 0 with branch cut on (-inf, 0).
    """
    if x.imag == 0 and x.real < 0:
        if not is_complex(x):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        x += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, x.imag)
    return mp.e1(x)


@overload
def exp10(x: Real) -> Real: ...
@overload
def exp10(x: Complex) -> Complex: ...


@reference_implementation()
def exp10(x):
    """Compute 10**x."""
    return mp.mpf(10) ** x


@overload
def exp2(x: Real) -> Real: ...
@overload
def exp2(x: Complex) -> Complex: ...


@reference_implementation()
def exp2(x):
    """Compute 2**x."""
    return mp.mpf(2) ** x


@overload
def expi(x: Real) -> Real: ...
@overload
def expi(x: Complex) -> Complex: ...


@reference_implementation()
def expi(x):
    """Exponential integral Ei.

    Notes
    -----
    Logarithmic singularity at x = 0 with branch cut on (-inf, 0).
    """
    if x.imag == 0 and x.real < 0:
        if not is_complex(x):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        x += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, x.imag)
    return mp.ei(x)


@reference_implementation()
def expit(x: Real) -> Real:
    """Expit (a.k.a logistic sigmoid)."""
    return mp.sigmoid(x)


@overload
def expm1(x: Real) -> Real: ...
@overload
def expm1(x: Complex) -> Complex: ...


@reference_implementation()
def expm1(x):
    """exp(x) - 1.

    Notes
    -----
    expm1 is an entire function
    """
    return mp.expm1(x)

@overload
def expn(n: Integer, x: Real) -> Real: ...
@overload
def expn(n: Real, x: Real) -> Real: ...

@reference_implementation()
def expn(n, x):
    """Generalized exponential integral En."""
    return mp.expint(n, x)


@reference_implementation()
def exprel(x: Real) -> Real:
    """Relative error exponential, (exp(x) - 1)/x."""
    if x == 0:
        return mp.one
    # set the precision high enough to avoid catastrophic cancellation
    precision = int(mp.ceil(-mp.log(abs(x), b=2))) + 53 if 0 < abs(x) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        # Near 0, mp.exp(x) - 1 = x + O(x^2)
        result = (mp.exp(x) - 1) / x
    return result


@reference_implementation()
def fdtr(dfn: Real, dfd: Real, x: Real) -> Real:
    """F cumulative distribution function."""
    if x < 0 or dfn < 0 or dfd < 0:
        return mp.nan
    x_dfn = x * dfn
    return betainc._mp(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn))


@reference_implementation()
def fdtrc(dfn: Real, dfd: Real, x: Real) -> Real:
    """F survival function."""
    if x < 0 or dfn < 0 or dfd < 0:
        return mp.nan
    x_dfn = x * dfn
    return betaincc._mp(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn))


@reference_implementation()
def fdtri(dfn: Real, dfd: Real, p: Real) -> Real:
    """F cumulative distribution function."""
    if p < 0 or p > 1 or dfn < 0 or dfd < 0:
        return mp.nan
    if p == 0:
        return 0.0
    if p == 1:
        return mp.inf
    q = betaincinv._mp(dfn / 2, dfd / 2, p)
    return q * dfd / ((1 - q) * dfn)

@overload
def fresnel(x: Real) -> Tuple[Real, Real]: ...
@overload
def fresnel(x: Complex) -> Tuple[Complex, Complex]: ...


@reference_implementation()
def fresnel(x):
    """Fresnel integrals.

    Notes
    -----
    Fresnel integrals are entire functions
    """
    return mp.fresnels(x), mp.fresnelc(x)


@overload
def gamma(x: Real) -> Real: ...
@overload
def gamma(x: Complex) -> Complex: ...


@reference_implementation()
def gamma(x):
    """Gamma function.

    Notes
    -----
    Poles at nonpositive integers
    """
    if x == 0.0:
        if isinstance(x, float):
            return math.copysign(mp.inf, x)
        return mp.nan
    if x.real < 0 and x.imag == 0 and x.real == mp.floor(x.real):
        return mp.nan
    return mp.gamma(x)


@reference_implementation()
def gammaincc(a: Real, x: Real) -> Real:
    """Regularized upper incomplete gamma function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a, x) > 1e6:
        return to_mp(special.gammainc(to_fp(a), to_fp(x)))
    try:
        return mp.gammainc(a, x, mp.inf, regularized=True)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.gammaincc(to_fp(a), to_fp(x)))


@reference_implementation()
def gammainc(a: Real, x: Real) -> Real:
    """Regularized lower incomplete gamma function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a, x) > 1e6:
        return to_mp(special.gammainc(to_fp(a), to_fp(x)))
    try:
        return mp.gammainc(a, 0, x, regularized=True)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.gammainc(to_fp(a), to_fp(x)))


@reference_implementation(uses_mp=False)
def gammainccinv(a: Real, y: Real) -> Real:
    """Inverse to the regularized upper incomplete gamma function."""
    return special.gammainccinv(a, y)


@reference_implementation(uses_mp=False)
def gammaincinv(a: Real, y: Real) -> Real:
    """Inverse to the regularized lower incomplete gamma function."""
    return special.gammaincinv(a, y)


@reference_implementation()
def gammaln(x: Real) -> Real:
    """Logarithm of the absolute value of the gamma function."""
    if x.real <= 0 and x == int(x):
        return mp.inf
    return mp.log(abs(mp.gamma(x)))


@reference_implementation()
def gammasgn(x: Real) -> Real:
    """Sign of the gamma function."""
    if x == 0.0:
        return math.copysign(1.0, x)
    if x == -mp.inf or x < 0 and x == int(x):
        return mp.nan
    return mp.sign(mp.gamma(x))


@reference_implementation()
def gdtr(a: Real, b: Real, x: Real) -> Real:
    """Gamma distribution cumulative distribution function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a*x, b) > 1e6:
        return to_mp(special.gdtr(to_fp(a),to_fp(b), to_fp(x)))
    try:
        # gammainc._mp is not used so that we can fall back to
        # special.gdtr instead of special.gammainc.
        return mp.gammainc(b, 0, a * x, regularized=True)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.gdtr(to_fp(a), to_fp(b), to_fp(x)))


@reference_implementation()
def gdtrc(a: Real, b: Real, x: Real)-> Real:
    """Gamma distribution survival function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a*x, b) > 1e6:
        return to_mp(special.gdtrc(to_fp(a), to_fp(b), to_fp(x)))
    try:
        # gammaincc._mp is not used so that we can fall back to
        # special.gdtrc instead of special.gammaincc.
        return mp.gammainc(b, a * x, mp.inf, regularized=True)
    except (mp.NoConvergence, ValueError):
        return to_mp(special.gdtrc(to_fp(a), to_fp(b), to_fp(x)))


@reference_implementation(uses_mp=False)
def gdtrib(a: Real, p: Real, x: Real) -> Real:
    """Inverse of `gdtr` vs b."""
    return special.gdtrib(a, p, x)


@overload
def hyp1f1(a: Real, b: Real, z: Real) -> Real: ...
@overload
def hyp1f1(a: Real, b: Real, z: Complex) -> Complex: ...


@reference_implementation()
def hyp1f1(a, b, z):
    """Confluent hypergeometric function 1F1.

    Notes
    -----
    Entire in a and z
    Meromorphic in b with poles at nonpositive integers
    """
    try:
        return mp.hyp1f1(a, b, z)
    except ZeroDivisionError:
        return mp.inf
    except Exception:
        return to_mp(special.hyp1f1(to_fp(a), to_fp(b), to_fp(z)))


@overload
def hyp2f1(a: Real, b: Real, c: Real, z: Real) -> Real: ...
@overload
def hyp2f1(a: Real, b: Real, c: Real, z: Complex) -> Complex: ...


@reference_implementation()
def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z).

    Notes
    -----
    Branch point at ``z=1`` with branch cut along ``(1, inf)``
    except for a or b a non-positive integer, in which case hyp2f1 reduces
    to a polynomial.
    """
    if z.imag == 0 and z.real > 1:
        if not (a == mp.floor(a) or b == mp.floor(b)):
            if not is_complex(z):
                # We will get a complex value on the branch cut, so if received real
                # input and expecting real output, just return NaN.
                return mp.nan
            # On branch cut, choose branch based on sign of zero
            z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)
    try:
        return mp.hyp2f1(a, b, c, z)
    except ZeroDivisionError:
        return mp.inf
    except Exception:
        # mpmath can't do it it, so fall back to SciPy 1.15.
        return to_mp(special.hyp2f1(to_fp(a), to_fp(b), to_fp(c), to_fp(z)))


@reference_implementation()
def hyperu(a: Real, b: Real, z: Real) -> Real:
    """Confluent hypergeometric function U.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(mp.one, z.imag)
    return mp.hyperu(a, b, z)


@reference_implementation()
def it1i0k0(x: Real) -> Tuple[Real, Real]:
    """Integrals of modified Bessel functions of order 0."""
    result1 = mp.quad(cyl_bessel_i0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_k0._mp, [0, x])
    return result1, result2


@reference_implementation()
def it1j0y0(x: Real) -> Tuple[Real, Real]:
    """Integrals of Bessel functions of the first kind of order 0."""
    result1 = mp.quad(cyl_bessel_j0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_y0._mp, [0, x])
    return result1, result2


@reference_implementation(uses_mp=False)
def it2i0k0(x: Real) -> Tuple[Real, Real]:
    """Integrals related to modified Bessel functions of order 0."""
    return special.it2i0k0(x)


@reference_implementation()
def it2j0y0(x: Real) -> Tuple[Real, Real]:
    """Integrals related to Bessel functions of the first kind of order 0."""

    def f1(t):
        return (1 - cyl_bessel_j0._mp(t)) / t

    def f2(t):
        return cyl_bessel_y0._mp(t) / t

    result1 = mp.quad(f1, [0, x])
    result2 = mp.quad(f2, [0, x])
    return result1, result2


@reference_implementation()
def it2struve0(x: Real) -> Real:
    """Integral related to the Struve function of order 0."""
    def f(t):
        return struve_h._mp(0, t) / t

    return mp.quad(f, [0, x])


@reference_implementation()
def itairy(x: Real) -> Tuple[Real, Real, Real, Real]:
    """Integrals of Airy functions."""
    def ai(t):
        return mp.airyai(t)

    def bi(t):
        return mp.airybi(t)

    result1 = mp.quad(ai, [0, x])
    result2 = mp.quad(bi, [0, x])
    result3 = mp.quad(ai, [-x, 0])
    result4 = mp.quad(bi, [-x, 0])
    return result1, result2, result3, result4


@reference_implementation()
def itmodstruve0(x: Real) -> Real:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation()
def itstruve0(x: Real) -> Real:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation()
def iv_ratio(v: Real, x: Real) -> Real:
    """Returns the ratio ``iv(v, x) / iv(v - 1, x)``"""
    numerator = cyl_bessel_i._mp(v, x)
    return numerator / cyl_bessel_i._mp(v - 1, x)


@reference_implementation()
def iv_ratio_c(v: Real, x: Real) -> Real:
    """Returns ``1 - iv_ratio(v, x)``."""
    numerator = cyl_bessel_i._mp(v, x)
    denominator = cyl_bessel_i._mp(v - 1, x)
    # Set precision high enough to avoid catastrophic cancellation.
    # For large x, iv_ratio_c(v, x) ~ (v - 0.5) / x
    if x != 0 and 0 < abs((v - 0.5) / x) < 1:
        precision = int(mp.ceil(-mp.log(abs((v - 0.5)/x), b=2))) + 53
        precision = max(mp.prec, precision)
    else:
        precision = mp.prec
    with mp.workprec(precision):
        result = mp.one - numerator / denominator
    return result


@reference_implementation()
def kei(x: Real) -> Real:
    """Kelvin function kei."""
    if x < 0:
        return mp.nan
    if x >= 1050:
        # kei increases monotonically to zero from below, can verify that
        # mp.kei(1050) == mpf('-1.3691040650084756e-324')
        # smaller than smallest subnormal in double precision.
        # Return something smaller than smallest subnormal so output processing
        # can pick up the correct sign of zero.
        return mp.mpf("-1e324")
    return mp.kei(0, x)


@reference_implementation()
def keip(x: Real) -> Real:
    """Derivative of the Kelvin function kei."""
    return to_mp(special.keip(to_fp(x)))


@reference_implementation()
def kelvin(x: Real) -> Tuple[Complex, Complex, Complex, Complex]:
    """Kelvin functions as complex numbers."""
    be = mp.mpc(ber._mp(x), bei._mp(x))
    ke = mp.mpc(ker._mp(x), kei._mp(x))
    bep = mp.mpc(berp._mp(x), beip._mp(x))
    kep = mp.mpc(kerp._mp(x), keip._mp(x))
    return be, ke, bep, kep


@reference_implementation()
def ker(x: Real) -> Real:
    """Kelvin function ker."""
    if x < 0:
        return mp.nan
    if x >= 1050:
        # ker decreases monotonically, can verify that
        # mp.ker(1050) == mpf('1.8167754471810517e-325')
        # smaller than smallest subnormal in double precision.
        return mp.zero
    return mp.ker(0, x)


@reference_implementation()
def kerp(x: Real) -> Real:
    """Derivative of the Kelvin function kerp."""
    return to_mp(special.kerp(to_fp(x)))


@reference_implementation(uses_mp=False)
def kolmogc(x: Real) -> Real:
    """CDF of Kolmogorov distribution.

    CDF of Kolmogorov distribution can be expressed in terms of
    Jacobi Theta functions.
    TODO: Look into writing arbitrary precision reference implementations
    for kolmogc, kolmogci, kolmogi, and kolmogorov.
    """
    return special._ufuncs._kolmogc(x)


@reference_implementation(uses_mp=False)
def kolmogci(x: Real) -> Real:
    """Inverse CDF of Kolmogorov distribution."""
    return special._ufuncs._kolmogci(x)


@reference_implementation(uses_mp=False)
def kolmogi(x: Real) -> Real:
    """Inverse Survival Function of Kolmogorov distribution."""
    return special.kolmogi(x)


@reference_implementation(uses_mp=False)
def kolmogorov(x: Real) -> Real:
    """Survival Function of Kolmogorov distribution."""
    return special.kolmogorov(x)


@reference_implementation(uses_mp=False)
def kolmogp(x: Real) -> Real:
    """Negative of PDF of Kolmogorov distribution.

    TODO: Why is this the negative pdf?
    """
    return special._ufuncs._kolmogp(x)


@reference_implementation()
def lambertw(z: Complex, k: Integer, eps: Real) -> Complex:
    """Lambert W function.

    Notes
    -----
    Branch cut on (-inf, 0) for ``k != 0``, (-inf, -1/e) for ``k = 0``
    ``k = 0`` corresponds to the principle branch.
    There are infinitely many branches.

    The tolerance eps is not actually used but included for compatibility.
    """
    if z.imag == 0 and (z.real < 0 and k !=0 or z.real < -1/mp.e):
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero.
        # mpmath's lambertw currently converts z to a complex128 internally,
        # so the small step here can't be smaller than the smallest subnormal.
        z += mp.mpc(0, 5e-324) * math.copysign(mp.one, z.imag)
    return mp.lambertw(z, k=k)


@reference_implementation()
def lanczos_sum_expg_scaled(z: Real) -> Real:
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    g = mp.mpf("6.024680040776729583740234375")
    return (mp.e / (z + g - 0.5)) ** (z - 0.5) * mp.gamma(z)


@reference_implementation()
def lgam1p(x: Real) -> Real:
    """Logarithm of abs(gamma(x + 1))."""
    if mp.isnan(x) or x == -mp.inf:
        return mp.nan
    if x == 0:
        return mp.zero
    if x == mp.inf:
        return mp.inf
    # set the precision high enough to resolve 1 + x.
    precision = int(mp.ceil(-mp.log(abs(x), b=2))) + 53 if 0 < abs(x) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        return gammaln._mp(x + 1)


@overload
def log1p(z: Real) -> Real: ...
@overload
def log1p(z: Complex) -> Complex: ...


@reference_implementation()
def log1p(z):
    """Logarithm of z + 1.

    Notes
    -----
    Branch cut on (-inf, -1)
    """
    if z.imag == 0 and z.real < -1:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(mp.one, z.imag)
    return mp.log1p(z)


@overload
def log1pmx(z: Real) -> Real: ...
@overload
def log1pmx(z: Complex) -> Complex: ...


@reference_implementation()
def log1pmx(z):
    """log(z + 1) - z.

    Notes
    -----
    Branch cut on (-inf, -1)
    """
    # set the precision high enough to avoid catastrophic cancellation.
    # Near z = 0 log(1 + z) - z = -z^2/2 + O(z^3)
    if z == 0:
        return mp.zero
    precision = (
        min(int(mp.ceil(-2*mp.log(abs(z), b=2))), 1024) + 53 if 0 < abs(z) < 1 else 0
    )
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = log1p._mp(z) - z
    return result


@overload
def loggamma(z: Real) -> Real: ...
@overload
def loggamma(z: Complex) -> Complex: ...


@reference_implementation()
def loggamma(z):
    """Principal branch of the logarithm of the gamma function.

    Notes
    -----
    Logarithmic singularity at z = 0
    Branch cut on (-inf, 0).
    Poles at nonnegative integers.
    """
    if z == 0:
        if isinstance(z, float):
            return math.copysign(mp.inf, z)
        return mp.nan
    if z.real < 0 and z.imag == 0 and z.real == mp.floor(z.real):
        return mp.nan
    if z.imag == 0 and z.real < 0:
        if not is_complex(z):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(mp.one, z.imag)
    return mp.loggamma(z)


@reference_implementation()
def log_expit(x: Real) -> Real:
    """Log of `expit`."""
    return mp.log(mp.sigmoid(x))


@reference_implementation(timeout=10)
def log_wright_bessel(a: Real, b: Real, x: Real) -> Real:
    """Natural logarithm of Wright's generalized Bessel function."""
    return mp.log(wright_bessel._mp(a, b, x))


@reference_implementation()
def logit(p: Real) -> Real:
    """Logit function ``logit(p) = log(p/(1 - p))``"""
    if p == 1:
        return mp.inf
    if p < 0 or p > 1:
        return mp.nan
    # set the precision high enough to resolve 1 - p.
    precision = int(mp.ceil(-mp.log(abs(p), b=2))) + 53 if 0 < abs(p) < 1 else 0
    with mp.workprec(max(mp.prec, precision)):
        result = mp.log(p/(1-p))
    return result


@reference_implementation(uses_mp=False)
def mcm1(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even modified Mathieu function of the first kind and its derivative."""
    return special.mathieu_modcem1(m, q, x)


@reference_implementation(uses_mp=False)
def mcm2(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even modified Mathieu function of the second kind and its derivative."""
    return special.mathieu_modcem2(m, q, x)


@reference_implementation(uses_mp=False)
def modified_fresnel_minus(x: Real) -> Tuple[Complex, Complex]:
    """Modified Fresnel negative integrals."""
    return special.modfresnelm(x)


@reference_implementation(uses_mp=False)
def modified_fresnel_plus(x: Real) -> Tuple[Complex, Complex]:
    """Modified Fresnel negative integrals."""
    return special.modfresnelp(x)


@reference_implementation(uses_mp=False)
def msm1(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd modified Mathieu function of the first kind and its derivative."""
    return special.mathieu_modsem1(m, q, x)


@reference_implementation(uses_mp=False)
def msm2(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd modified Mathieu function of the second kind and its derivative."""
    return special.mathieu_modsem2(m, q, x)


@overload
def nbdtr(k: Integer, n: Integer, p: Real) -> Real: ...
@overload
def nbdtr(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation()
def nbdtr(k, n, p):
    """Negative binomial cumulative distribution function."""
    if (
            mp.isinf(k) or mp.isinf(n)
            or p > 1 or p < 0 or k < 0 or n < 0
    ):
        return mp.nan
    return betainc._mp(n, k + 1, p)


@overload
def nbdtrc(k: Integer, n: Integer, p: Real) -> Real: ...
@overload
def nbdtrc(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation()
def nbdtrc(k, n, p):
    """Negative binomial survival function."""
    if (
            mp.isinf(k) or mp.isinf(n)
            or p > 1 or p < 0 or k < 0 or n < 0
    ):
        return mp.nan
    return betaincc._mp(n, k + 1, p)


@overload
def ndtr(x: Real) -> Real: ...
@overload
def ndtr(x: Complex) -> Complex: ...


@reference_implementation()
def ndtr(x):
    """Cumulative distribution of the standard normal distribution."""
    if x.imag == 0:
        return mp.ncdf(x.real)
    return (1 + erf._mp(x/mp.sqrt(2)))/2


@reference_implementation()
def ndtri(y: Real) -> Real:
    """Inverse of `ndtr` vs x."""
    if not 0 <= y <= 1:
        return mp.nan
    if y == 0:
        return -mp.inf
    if y == 1:
        return mp.inf
    # set the precision high enough to resolve 2*y - 1
    precision = int(mp.ceil(-mp.log(abs(2*y), b=2))) + 53 if 0 < abs(2*y) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        # set the precision high enough to resolve 2*y - 1
        result = mp.sqrt(2) * mp.erfinv(2*y - 1)
    return result


@reference_implementation(uses_mp=False)
def oblate_aswfa(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function obl_ang1 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_ang1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_aswfa_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function of the first kind and its derivative."""
    return special.obl_ang1(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_radial1(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function obl_rad1 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_rad1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_radial1_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function of the first kind and its derivative."""
    return special.obl_rad1(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_radial2(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function obl_rad2 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_rad2_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_radial2_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function of the second kind and its derivative."""
    return special.obl_rad2(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_segv(m: Real, n: Real, c: Real) -> Real:
    """Characteristic value of oblate spheroidal function."""
    return special.obl_cv(m, n, c)


@reference_implementation()
def owens_t(h: Real, a: Real) -> Real:
    """Owen's T Function."""
    def integrand(x):
        return mp.exp(-(h**2) * (1 + x**2) / 2) / (1 + x**2)

    return mp.quad(integrand, [0, a]) / (2 * mp.pi)


@reference_implementation()
def pbdv(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function D."""
    d = mp.pcfd(v, x)
    _, dp = special.pbdv(to_fp(v), to_fp(x))
    return d, to_mp(dp)


@reference_implementation()
def pbvv(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function V."""
    # Set precision to guarantee -v - 0.5 retains precision for very small v.
    precision = int(mp.ceil(-mp.log(abs(2*v), b=2))) + 53 if 0 < abs(2*v) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        d = mp.pcfv(-v - 0.5, x)
    _, dp = special.pbvv(to_fp(v), to_fp(x))
    return d, to_mp(dp)


@reference_implementation()
def pbwa(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function W."""
    d = mp.pcfw(v, x)
    _, dp = special.pbwa(to_fp(v), to_fp(x))
    return d, to_mp(dp)


@reference_implementation()
def pdtr(k: Real, m: Real) -> Real:
    """Poisson cumulative distribution function."""
    if k < 0 or m < 0:
        return mp.nan
    k = mp.floor(k)
    return gammaincc._mp(k + 1, m)


@reference_implementation()
def pdtrc(k: Real, m: Real) -> Real:
    """Poisson survival function."""
    if k < 0 or m < 0:
        return mp.nan
    k = mp.floor(k)
    return gammainc._mp(k + 1, m)


@overload
def pdtri(k: Integer, y: Real) -> Real: ...
@overload
def pdtri(k: Real, y: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def pdtri(k, y):
    """Inverse of `pdtr` vs m."""
    return special.pdtri(k, y)


@reference_implementation(uses_mp=False)
def pmv(m: Integer, v: Real, x: Real) -> Real:
    """Associated Legendre function of integer order and real degree."""
    return special.lpmv(m, v, x)


@reference_implementation()
def poch(z: Real, m: Real) -> Real:
    """Pochhammer symbol."""
    return mp.rf(z, m)


@reference_implementation(uses_mp=False)
def prolate_aswfa(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function pro_ang1 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_ang1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_aswfa_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function of the first kind and its derivative."""
    return special.pro_ang1(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_radial1(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function pro_rad1 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_rad1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_radial1_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function of the first kind and its derivative."""
    return special.pro_rad1(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_radial2(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function pro_rad2 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_rad2_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_radial2_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function of the second kind and its derivative."""
    return special.pro_rad2(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_segv(m: Real, n: Real, c: Real) -> Real:
    """Characteristic value of prolate spheroidal function."""
    return special.pro_cv(m, n, c)


@reference_implementation()
def radian(d: Real, m: Real, s: Real) -> Real:
    """Convert from degrees to radians."""
    return mp.radians(d + m / 60 + s / 3600)


@overload
def rgamma(z: Real) -> Real: ...
@overload
def rgamma(z: Complex) -> Complex: ...


@reference_implementation()
def rgamma(z):
    """Reciprocal of the gamma function."""
    if z == 0.0:
        return z
    if z.imag == 0 and z.real < 0 and z.real == mp.floor(z.real):
        return 0.0
    return mp.one / mp.gamma(z)


@overload
def riemann_zeta(z: Real) -> Real: ...
@overload
def riemann_zeta(z: Complex) -> Complex: ...


@reference_implementation()
def riemann_zeta(z):
    """Riemann zeta function.

    Notes
    -----
    A single pole at z = 1
    """
    if z == 1.0:
        return mp.nan
    return mp.zeta(z)


@reference_implementation()
def round(x: Real) -> Real:
    """Round to the nearest integer."""
    return mp.nint(x)


@reference_implementation()
def scaled_exp1(x: Real) -> Real:
    """Exponentially scaled exponential integral E1."""
    return mp.exp(x) * x * mp.e1(x)


@reference_implementation(uses_mp=False)
def sem(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd Mathieu function and its derivative."""
    return special.mathieu_sem(m, q, x)


@reference_implementation(uses_mp=False)
def sem_cva(m: Real, q: Real) -> Real:
    """Characteristic value of odd Mathieu functions."""
    return special.mathieu_b(m, q)


@overload
def shichi(x: Real) -> Tuple[Real, Real]: ...
@overload
def shichi(x: Complex) -> Tuple[Complex, Complex]: ...


@reference_implementation()
def shichi(x):
    """Hyperbolic sine and cosine integrals.

    Notes
    -----
    Hyperbolic sine and cosine integrals are entire functions

    """
    return mp.shi(x), mp.chi(x)


@overload
def sici(x: Real) -> Tuple[Real, Real]: ...
@overload
def sici(x: Complex) -> Tuple[Complex, Complex]: ...


@reference_implementation()
def sici(x):
    """Sine and cosine integrals.

    Notes
    -----
    Sine and cosine integrals are entire functions
    """
    return mp.si(x), mp.ci(x)


@reference_implementation(uses_mp=False)
def sindg(x):
    """Sine of the angle `x` given in degrees."""
    return special.sindg(x)


@overload
def sinpi(x: Real) -> Real: ...
@overload
def sinpi(x: Complex) -> Complex: ...


@reference_implementation()
def sinpi(x):
    """Sine of pi*x.

    Note
    ----
    sinpi is an entire function
    """
    if x.imag == 0 and x.real == mp.floor(x.real):
        # Something smaller than smallest subnormal will be converted to a zero
        # of the correct sign by output processing.
        return math.copysign(mp.mpf("1e-324"), x.real)
    return mp.sinpi(x)


@overload
def smirnov(n: Integer, p: Real) -> Real: ...
@overload
def smirnov(n: Real, p: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def smirnov(n, d):
    """Kolmogorov-Smirnov complementary cumulative distribution function."""
    return special.smirnov(n, d)


@overload
def smirnovc(n: Integer, p: Real) -> Real: ...
@overload
def smirnovc(n: Real, p: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def smirnovc(n, d):
    """Kolmogorov-Smirnov cumulative distribution function."""
    return special._ufuncs._smirnovc(n, d)


@overload
def smirnovci(n: Integer, p: Real) -> Real: ...
@overload
def smirnovci(n: Real, p: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def smirnovci(n, p):
    """Inverse to `smirnovc`."""
    return special._ufuncs._smirnovci(n, p)


@overload
def smirnovi(n: Integer, p: Real) -> Real: ...
@overload
def smirnovi(n: Real, p: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def smirnovi(n, p):
    """Inverse to `smirnov`."""
    return special.smirnovi(n, p)


@overload
def smirnovp(n: Integer, d: Real) -> Real: ...
@overload
def smirnovp(n: Real, d: Real) -> Real: ...


@reference_implementation(uses_mp=False)
def smirnovp(n, d):
    """Negative of Kolmogorov-Smirnov pdf."""
    return special._ufuncs._smirnovp(n, d)


@overload
def spence(z: Real) -> Real: ...
@overload
def spence(z: Complex) -> Complex: ...


@reference_implementation()
def spence(z):
    """Spence's function, also known as the dilogarithm."""
    # set the precision high enough that mp.one - z != 1
    precision = int(mp.ceil(-mp.log(abs(z), b=2))) + 53 if 0 < abs(z) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = mp.polylog(2, mp.one - z)
    return result


@reference_implementation()
def struve_h(v: Real, x: Real) -> Real:
    """Struve function."""
    return mp.struveh(v, x)


@reference_implementation()
def struve_l(v: Real, x: Real) -> Real:
    """Modified Struve function."""
    return  mp.struvel(v, x)


@reference_implementation(uses_mp=False)
def tandg(x: Real) -> Real:
    """Tangent of angle x given in degrees."""
    return special.tandg(x)


@reference_implementation(uses_mp=False)
def voigt_profile(x: Real, sigma: Real, gamma: Real) -> Real:
    """Voigt profile"""
    return special.voigt_profile(x, sigma, gamma)


@overload
def wofz(x: Real) -> Real: ...
@overload
def wofz(x: Complex) -> Complex: ...


@reference_implementation()
def wofz(x):
    """Faddeeva function

    Notes
    -----
    wofz is an entire function
    """
    return mp.exp(-x**2) * mp.erfc(-mp.j * x)


@reference_implementation(timeout=10)
def wright_bessel(a: Real, b: Real, x: Real) -> Real:
    """Wright's generalized Bessel function."""
    def term(k):
        m = a * k + b
        if m <= 0 and m == mp.floor(m):
            return mp.zero
        return x**k / (mp.factorial(k) * mp.gamma(a * k + b))

    return mp.nsum(term, [0, mp.inf])


@overload
def xlogy(x: Real, y: Real) -> Real: ...
@overload
def xlogy(x: Complex, y: Complex) -> Complex: ...


@reference_implementation()
def xlogy(x, y):
    """Compute ``x*log(y)`` so that the result is 0 if ``x = 0``."""
    if x == 0 and not (mp.isnan(x.real) or mp.isnan(x.imag)):
        return 0
    if y.imag == 0 and y.real < 0:
        if not is_complex(y):
            # We will get a complex value on the branch cut, so if received real
            # input and expecting real output, just return NaN.
            return mp.nan
        # On branch cut, choose branch based on sign of zero.
        y += mp.mpc(0, "1e-1000000000") * math.copysign(mp.one, y.imag)
    return x * mp.log(y)


@overload
def xlog1py(x: Real, y: Real) -> Real: ...
@overload
def xlog1py(x: Complex, y: Complex) -> Complex: ...


@reference_implementation()
def xlog1py(x, y):
    """Compute ``x*log(y)`` so that the result is 0 if ``x = 0``."""
    if x == 0 and not (mp.isnan(x.real) or mp.isnan(x.imag)):
        return mp.zero
    return x * log1p._mp(y)


@reference_implementation()
def zeta(z: Real, q: Real) -> Real:
    """Hurwitz zeta function."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z, a=q)


@reference_implementation()
def zetac(z: Real) -> Real:
    """Riemann zeta function minus 1."""
    # set the precision high enough to avoid catastrophic cancellation.
    # As z approaches +inf in the right halfplane:
    # zeta(z) - 1 = 2^-z + O(3^-z).
    if z == 1:
        return mp.nan
    precision = int(mp.ceil(z.real)) + 53 if abs(z.real) < 1 else 0
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = mp.zeta(z) - mp.one
    return result


def solve_bisect(f, xl, xr, *, maxiter=1000):
    if not xl < xr:
        xl, xr = xr, xl
    fl, fr = f(xl), f(xr)
    if fl == 0:
        return xl
    if fr == 0:
        return xr
    if mp.sign(fl) == mp.sign(fr):
        raise ValueError("f(xl) and f(xr) must have different signs")

    DBL_MAX = sys.float_info.max
    DBL_TRUE_MIN = 5e-324

    # Special handling for case where initial interval contains 0. It
    # can take a long time to find a root near zero to a given
    # relative tolerance through bisection alone, so this makes an
    # effort to find a better starting bracket.
    if xl <= 0 <= xr:
        f0 = f(0)
        if f0 == 0:
            return mp.zero
        vals = np.asarray([1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300, 5e-324])
        if mp.sign(f0) == mp.sign(fr):
            vals = -vals
            for t in vals:
                if xl > t:
                    continue
                ft = f(t)
                if  mp.sign(ft) != mp.sign(fl):
                    xr = t;
                    break
                xl = t
        else:
            for t in vals:
                if xr < t:
                    continue
                ft = f(t)
                if  mp.sign(ft) != mp.sign(fr):
                    xl = t;
                    break
                xr = t

    iterations = Bisection(mp, f, [xl, xr])
    x_prev = mp.inf
    for i, (x, error) in enumerate(iterations):
        if i == maxiter:
            raise RuntimeError("maxiter exceeded")
        if abs(x - x_prev) < abs(x)*1e-17:
            break
        if x < DBL_TRUE_MIN:
            return mp.zero
        if x > DBL_MAX:
            return mp.inf
        x_prev = x
    return x


_exclude = [
    "is_complex"
    "math",
    "mp",
    "np",
    "overload",
    "reference_implementation",
    "scipy",
    "solve_bisect",
    "special",
    "sys",
    "to_fp",
    "to_mp",
    "version",
    "Bisection",
    "Complex",
    "Integer",
    "Real",
    "Secant",
    "Tuple",
]

__all__ = [s for s in dir() if not s.startswith("_") and s not in _exclude]
