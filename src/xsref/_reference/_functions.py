import math
import numpy as np
import scipy.special._ufuncs as ufuncs
import sys

from mpmath import mp  # type: ignore
from mpmath.calculus.optimization import Bisection, Secant
from packaging import version
from typing import overload, Tuple

from numpy import integer as Integer
from numpy import floating as Real
from numpy import complexfloating as Complex

from ._framework import reference_implementation


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


def get_resolution_precision(*, x=None, log2abs_x=None):
    """Identify the level of precision needed to resolve 1 + x.

    This is the precision level needed so that
    ``float((1 + x) - 1)`` will recover `x` with no loss of precision
    due to catastrophic cancellation.
    """
    if x is not None and log_abs_x is not None:
        raise ValueError
    if x is not None:
        if x == 0 or not mp.isfinite(x):
            return mp.prec
        log2abs_x = mp.log(abs(x), b=2)
    # 1075 is the precision needed to resolve the smallest subnormal.
    return min(
        max(int(mp.ceil(-log2abs_x)) + 53, mp.prec), 1075
    )


@overload
def airy(x: Real) -> Tuple[Real, Real, Real, Real]: ...
@overload
def airy(x: Complex) -> Tuple[Complex, Complex, Complex, Complex]: ...


@reference_implementation(scipy=ufuncs.airy)
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


@reference_implementation(scipy=ufuncs.airye)
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


@reference_implementation(scipy=ufuncs.bdtr)
def bdtr(k, n, p):
    """Binomial distribution cumulative distribution function."""
    k, n = mp.floor(k), mp.floor(n)
    if p > 1 or p < 0 or k < 0 or n < 0 or k > n or mp.isinf(n):
        return mp.nan
    if k == n:
        return mp.one
    # set the precision high enough that mp.one - p != 1
    with mp.workprec(get_resolution_precision(x=p)):
        result = betainc._mp(n - k, k + 1, 1 - p)
    return result


@overload
def bdtrc(k: Real, n: Integer, p: Real) -> Real: ...
@overload
def bdtrc(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation(scipy=ufuncs.bdtrc)
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


@reference_implementation(scipy=ufuncs.bdtri)
def bdtri(k, n, y):
    if y > 1 or y < 0 or k < 0 or n < 0 or k > n or mp.isinf(n):
        return mp.nan
    """Inverse function to `bdtr` with respect to `p`."""
    k, n = mp.floor(k), mp.floor(n)

    def f(p):
        return bdtr._mp(k, n, p) - y

    return solve_bisect(f, 0, 1)


@reference_implementation(scipy=ufuncs.bei)
def bei(x: Real) -> Real:
    """Kelvin function bei."""
    return mp.bei(0, x)


@reference_implementation(scipy=ufuncs.beip)
def beip(x: Real) -> Real:
    """Derivative of the Kelvin function bei."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.ber)
def ber(x: Real) -> Real:
    """Kelvin function ber."""
    return mp.ber(0, x)


@reference_implementation(scipy=ufuncs.berp)
def berp(x: Real) -> Real:
    """Derivative of the Kelvin function ber."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.besselpoly)
def besselpoly(a: Real, lmb: Real, nu: Real) -> Real:
    """Weighted integral of the Bessel function of the first kind."""
    def integrand(x):
        return x**lmb * cyl_bessel_j._mp(nu, 2 * a * x)

    return mp.quad(integrand, [0, 1])


@reference_implementation(scipy=ufuncs.beta)
def beta(a: Real, b: Real) -> Real:
    """Beta function."""
    return mp.beta(a, b)


@reference_implementation(scipy=ufuncs.betaln)
def betaln(a: Real, b: Real) -> Real:
    """Natural logarithm of the absolute value of the Beta function."""
    return mp.log(abs(mp.beta(a, b)))


@reference_implementation(scipy=ufuncs.betainc)
def betainc(a: Real, b: Real, x: Real) -> Real:
    """Regularized incomplete Beta function."""
    return mp.betainc(a, b, 0, x, regularized=True)


@reference_implementation(scipy=ufuncs.betaincc)
def betaincc(a: Real, b: Real, x: Real) -> Real:
    """Complement of the regularized incomplete Beta function."""
    return mp.betainc(a, b, x, 1.0, regularized=True)


@reference_implementation(scipy=ufuncs.betaincinv)
def betaincinv(a: Real, b: Real, y: Real) -> Real:
    """Inverse of the regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, 0, x, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation(scipy=ufuncs.betainccinv)
def betainccinv(a: Real, b: Real, y: Real) -> Real:
    """Inverse of the complemented regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, x, 1.0, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation(scipy=ufuncs.binom)
def binom(n: Real, k: Real) -> Real:
    """Binomial coefficient considered as a function of two real variables."""
    return mp.binomial(n, k)


@reference_implementation(scipy=ufuncs.cbrt)
def cbrt(x: Real) -> Real:
    """Cube root of x."""
    return mp.cbrt(x)


@reference_implementation(scipy=ufuncs.mathieu_cem)
def cem(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even Mathieu function and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.mathieu_a)
def cem_cva(m: Real, q: Real) -> Real:
    """Characteristic value of even Mathieu functions."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.chdtr)
def chdtr(v: Real, x: Real) -> Real:
    """Chi square cumulative distribution function."""
    if x < 0 or v < 0:
        return mp.nan
    return gammainc._mp(v / 2, x / 2)

@reference_implementation(scipy=ufuncs.chdtrc)
def chdtrc(v: Real, x: Real) -> Real:
    """Chi square survival function."""
    if x < 0 or v < 0:
        return mp.nan
    return gammaincc._mp(v / 2, x / 2)


@reference_implementation(scipy=ufuncs.chdtri)
def chdtri(v: Real, p: Real) -> Real:
    """Inverse to `chdtrc` with respect to `x`."""
    # TODO. Figure out why chdtri inverts chdtrc and not chdtr
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.cosdg)
def cosdg(x: Real) -> Real:
    """Cosine of the angle x given in degrees."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.cosm1)
def cosm1(x: Real) -> Real:
    """cos(x) - 1 for use when x is near zero."""
    # set the precision high enough to avoid catastrophic cancellation
    # cos(x) - 1 = x^2/2 + O(x^4) for x near 0
    if not mp.isfinite(x):
        return mp.nan
    with mp.workprec(get_resolution_precision(log2abs_x=2*mp.log(abs(x/2), b=2))):
        result =  mp.cos(x) - mp.one
    return result


@overload
def cospi(x: Real) -> Real: ...
@overload
def cospi(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs._cospi)
def cospi(x):
    """Cosine of pi*x."""
    # This already does the right thing regarding sign of zero for the case
    # x - 0.5 an integer.
    return mp.cospi(x)


@reference_implementation(scipy=ufuncs.cotdg)
def cotdg(x: Real) -> Real:
    """Cotangent of the angle x given in degrees."""
    raise NotImplementedError


@overload
def cyl_bessel_i(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_i(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.iv)
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
        raise NotImplementedError

    result = mp.besseli(v, z)

    if z.imag == 0 and z.real < 0 and v == mp.floor(v):
        # No discontinuity for integer v and should return a real value.
        # Numerical error can cause a small imaginary part here, so just return
        # the real part.
        result = result.real
    return result


@reference_implementation(scipy=ufuncs.i0)
def cyl_bessel_i0(z: Real) -> Real:
    """Modified Bessel function of order 0."""
    return cyl_bessel_i._mp(0, z)


@reference_implementation(scipy=ufuncs.i0e)
def cyl_bessel_i0e(z: Real) -> Real:
    """Exponentially scaled modified Bessel function of order 0."""
    return mp.exp(-abs(z.real)) * cyl_bessel_i0._mp(z)


@reference_implementation(scipy=ufuncs.i1)
def cyl_bessel_i1(z: Real)-> Real:
    """Modified Bessel function of order 1."""
    return cyl_bessel_i._mp(1, z)


@reference_implementation(scipy=ufuncs.i1e)
def cyl_bessel_i1e(z: Real) -> Real:
    """Exponentially scaled modified Bessel function of order 1."""
    return mp.exp(-abs(z.real)) * cyl_bessel_i1._mp(z)


@overload
def cyl_bessel_ie(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ie(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.ive)
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


@reference_implementation(scipy=ufuncs.jv)
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
        raise NotImplementedError

    result = prefactor * mp.besselj(v, z)

    if z.imag == 0 and z.real < 0 and v == mp.floor(v):
        # No discontinuity for integer v and should return a real value.
        # Numerical error can cause a small imaginary part here, so just return
        # the real part.
        result = result.real
    return result


@reference_implementation(scipy=ufuncs.j0)
def cyl_bessel_j0(x: Real) -> Real:
    """Bessel function of the first kind of order 0."""
    return mp.j0(x)


@reference_implementation(scipy=ufuncs.j1)
def cyl_bessel_j1(x: Real) -> Real:
    """Bessel function of the first kind of order 1."""
    return mp.j1(x)


@overload
def cyl_bessel_je(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_je(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.jve)
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


@reference_implementation(scipy=ufuncs.kv)
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
    return mp.besselk(v, z)


@reference_implementation(scipy=ufuncs.k0)
def cyl_bessel_k0(x: Real) -> Real:
    """Modified Bessel function of the second kinf of order 0."""
    return cyl_bessel_k._mp(0, x)


@reference_implementation(scipy=ufuncs.k0e)
def cyl_bessel_k0e(x: Real) -> Real:
    """Exponentially scaled modified Bessel function K of order 0."""
    return mp.exp(x) * cyl_bessel_k0._mp(x)


@reference_implementation(scipy=ufuncs.k1)
def cyl_bessel_k1(x: Real) -> Real:
    """Modified Bessel function of the second kind of order 0."""
    return cyl_bessel_k._mp(1, x)


@reference_implementation(scipy=ufuncs.k1e)
def cyl_bessel_k1e(x: Real) -> Real:
    """Exponentially scaled modified Bessel function K of order 1."""
    return mp.exp(x) * cyl_bessel_k1._mp(x)


@overload
def cyl_bessel_ke(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ke(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.kve)
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


@reference_implementation(scipy=ufuncs.yv, default_timeout=10)
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
        raise NotImplementedError

    return prefactor * mp.bessely(v, z)


@reference_implementation(scipy=ufuncs.y0)
def cyl_bessel_y0(z: Real) -> Real:
    """Bessel function of the second kind of order 0."""
    return cyl_bessel_y._mp(0, z)


@reference_implementation(scipy=ufuncs.y1)
def cyl_bessel_y1(z: Real) -> Real:
    """Bessel function of the second kind of order 1."""
    return cyl_bessel_y._mp(1, z)


@overload
def cyl_bessel_ye(v: Real, z: Real) -> Real: ...
@overload
def cyl_bessel_ye(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.yve)
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


@reference_implementation(scipy=ufuncs.hankel1)
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
    return mp.hankel1(v, z)


@overload
def cyl_hankel_1e(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_1e(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.hankel1e)
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


@reference_implementation(scipy=ufuncs.hankel2)
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
    return mp.hankel2(v, z)


@overload
def cyl_hankel_2e(v: Real, z: Real) -> Real: ...
@overload
def cyl_hankel_2e(v: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.hankel2e)
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


@reference_implementation(scipy=ufuncs.dawsn)
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


@reference_implementation(scipy=ufuncs.psi)
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


@reference_implementation(scipy=ufuncs.ellipe)
def ellipe(m: Real) -> Real:
    """Complete elliptic integral of the second kind."""
    return mp.ellipe(m)


@reference_implementation(scipy=ufuncs.ellipeinc)
def ellipeinc(phi: Real, m: Real) -> Real:
    """Incomplete elliptic integral of the second kind."""
    if not mp.isfinite(phi) or mp.isnan(m) or m == mp.inf:
        # mpmath doesn't handle these cases, use self reference for now.
        raise NotImplementedError
    return mp.ellipe(phi, m)


@reference_implementation(scipy=ufuncs.ellipj)
def ellipj(u: Real, m: Real) -> Tuple[Real, Real, Real, Real]:
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return sn, cn, dn, phi


@reference_implementation(scipy=ufuncs.ellipk)
def ellipk(m: Real) -> Real:
    """Complete elliptic integral of the first kind."""
    return mp.ellipk(m)


@reference_implementation(scipy=ufuncs.ellipkinc)
def ellipkinc(phi: Real, m: Real) -> Real:
    """Incomplete elliptic integral of the first kind."""
    if not mp.isfinite(phi) or mp.isnan(m) or m == mp.inf:
        # mpmath doesn't handle these cases, use self reference for now.
        raise NotImplementedError
    return mp.ellipf(phi, m)


@reference_implementation(scipy=ufuncs.ellipkm1)
def ellipkm1(p: Real) -> Real:
    """Complete elliptic integral of the first kind around m = 1."""
    # set the precision high enough to resolve 1 - p
    with mp.workprec(get_resolution_precision(x=p)):
        result = ellipk._mp(1 - p)
    return result


@overload
def erf(x: Real) -> Real: ...
@overload
def erf(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.erf)
def erf(x):
    """Error function.

    erf is an entire function
    """
    return mp.erf(x)


@overload
def erfi(x: Real) -> Real: ...
@overload
def erfi(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.erfi)
def erfi(x):
    """Imaginary error function.

    erfi is an entire function
    """
    return -mp.j * mp.erf(mp.j * x)


@overload
def erfc(x: Real) -> Real: ...
@overload
def erfc(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.erfc)
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


@reference_implementation(scipy=ufuncs.erfcx)
def erfcx(x):
    """Scaled complementary error function exp(x**2) * erfc(x)

    Notes
    -----
    erfcx is an entire function
    """
    return mp.exp(x**2) * mp.erfc(x)


@reference_implementation(scipy=ufuncs.erfcinv)
def erfcinv(x: Real) -> Real:
    """Inverse of the complementary error function."""
    if not 0 <= x <= 2:
        return mp.nan
    if x == 0:
        return mp.inf
    with mp.workprec(get_resolution_precision(x=p)):
        result = mp.erfinv(mp.one - x)
    return result


@overload
def exp1(x: Real) -> Real: ...
@overload
def exp1(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.exp1)
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


@reference_implementation(scipy=ufuncs.exp10)
def exp10(x):
    """Compute 10**x."""
    return mp.mpf(10) ** x


@overload
def exp2(x: Real) -> Real: ...
@overload
def exp2(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.exp2)
def exp2(x):
    """Compute 2**x."""
    return mp.mpf(2) ** x


@overload
def expi(x: Real) -> Real: ...
@overload
def expi(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.expi)
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


@reference_implementation(scipy=ufuncs.expit)
def expit(x: Real) -> Real:
    """Expit (a.k.a logistic sigmoid)."""
    return mp.sigmoid(x)


@overload
def expm1(x: Real) -> Real: ...
@overload
def expm1(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.expm1)
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

@reference_implementation(scipy=ufuncs.expn)
def expn(n, x):
    """Generalized exponential integral En."""
    return mp.expint(n, x)


@reference_implementation(scipy=ufuncs.exprel)
def exprel(x: Real) -> Real:
    """Relative error exponential, (exp(x) - 1)/x."""
    if x == 0:
        return mp.one
    # set the precision high enough to avoid catastrophic cancellation
    # Near 0, mp.exp(x) - 1 = x + O(x^2)
    with mp.workprec(get_resolution_precision(x=x)):
        result = (mp.exp(x) - 1) / x
    return result


@reference_implementation(scipy=ufuncs.fdtr)
def fdtr(dfn: Real, dfd: Real, x: Real) -> Real:
    """F cumulative distribution function."""
    if x < 0 or dfn < 0 or dfd < 0:
        return mp.nan
    x_dfn = x * dfn
    return betainc._mp(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn))


@reference_implementation(scipy=ufuncs.fdtrc)
def fdtrc(dfn: Real, dfd: Real, x: Real) -> Real:
    """F survival function."""
    if x < 0 or dfn < 0 or dfd < 0:
        return mp.nan
    x_dfn = x * dfn
    return betaincc._mp(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn))


@reference_implementation(scipy=ufuncs.fdtri)
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


@reference_implementation(scipy=ufuncs.fresnel)
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


@reference_implementation(scipy=ufuncs.gamma)
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


@reference_implementation(scipy=ufuncs.gammaincc)
def gammaincc(a: Real, x: Real) -> Real:
    """Regularized upper incomplete gamma function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a, x) > 1e6:
        raise NotImplementedError
    return mp.gammainc(a, x, mp.inf, regularized=True)


@reference_implementation(scipy=ufuncs.gammainc)
def gammainc(a: Real, x: Real) -> Real:
    """Regularized lower incomplete gamma function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a, x) > 1e6:
        raise NotImplementedError
    return mp.gammainc(a, 0, x, regularized=True)


@reference_implementation(scipy=ufuncs.gammainccinv)
def gammainccinv(a: Real, y: Real) -> Real:
    """Inverse to the regularized upper incomplete gamma function."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.gammaincinv)
def gammaincinv(a: Real, y: Real) -> Real:
    """Inverse to the regularized lower incomplete gamma function."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.gammaln)
def gammaln(x: Real) -> Real:
    """Logarithm of the absolute value of the gamma function."""
    if x.real <= 0 and x == int(x):
        return mp.inf
    return mp.log(abs(mp.gamma(x)))


@reference_implementation(scipy=ufuncs.gammasgn)
def gammasgn(x: Real) -> Real:
    """Sign of the gamma function."""
    if x == 0.0:
        return math.copysign(1.0, x)
    if x == -mp.inf or x < 0 and x == int(x):
        return mp.nan
    return mp.sign(mp.gamma(x))


@reference_implementation(scipy=ufuncs.gdtr)
def gdtr(a: Real, b: Real, x: Real) -> Real:
    """Gamma distribution cumulative distribution function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a*x, b) > 1e6:
        raise NotImplementedError
    return gammainc._mp(b, a * x)


@reference_implementation(scipy=ufuncs.gdtrc)
def gdtrc(a: Real, b: Real, x: Real)-> Real:
    """Gamma distribution survival function."""
    if a < 0 or x < 0:
        return mp.nan
    if min(a*x, b) > 1e6:
        raise NotImplementedError
    return gammaincc._mp(b, a * x)


@reference_implementation(scipy=ufuncs.gdtrib)
def gdtrib(a: Real, p: Real, x: Real) -> Real:
    """Inverse of `gdtr` vs b."""
    raise NotImplementedError


@overload
def hyp1f1(a: Real, b: Real, z: Real) -> Real: ...
@overload
def hyp1f1(a: Real, b: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.hyp1f1)
def hyp1f1(a, b, z):
    """Confluent hypergeometric function 1F1.

    Notes
    -----
    Entire in a and z
    Meromorphic in b with poles at nonpositive integers
    """
    return mp.hyp1f1(a, b, z)


@overload
def hyp2f1(a: Real, b: Real, c: Real, z: Real) -> Real: ...
@overload
def hyp2f1(a: Real, b: Real, c: Real, z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.hyp2f1)
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
    return mp.hyp2f1(a, b, c, z)


@reference_implementation(scipy=ufuncs.hyperu)
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


@reference_implementation(scipy=ufuncs.iti0k0)
def it1i0k0(x: Real) -> Tuple[Real, Real]:
    """Integrals of modified Bessel functions of order 0."""
    result1 = mp.quad(cyl_bessel_i0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_k0._mp, [0, x])
    return result1, result2


@reference_implementation(scipy=ufuncs.itj0y0)
def it1j0y0(x: Real) -> Tuple[Real, Real]:
    """Integrals of Bessel functions of the first kind of order 0."""
    result1 = mp.quad(cyl_bessel_j0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_y0._mp, [0, x])
    return result1, result2


@reference_implementation(scipy=ufuncs.it2i0k0)
def it2i0k0(x: Real) -> Tuple[Real, Real]:
    """Integrals related to modified Bessel functions of order 0."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.it2j0y0)
def it2j0y0(x: Real) -> Tuple[Real, Real]:
    """Integrals related to Bessel functions of the first kind of order 0."""

    def f1(t):
        return (1 - cyl_bessel_j0._mp(t)) / t

    def f2(t):
        return cyl_bessel_y0._mp(t) / t

    result1 = mp.quad(f1, [0, x])
    result2 = mp.quad(f2, [0, x])
    return result1, result2


@reference_implementation(scipy=ufuncs.it2struve0)
def it2struve0(x: Real) -> Real:
    """Integral related to the Struve function of order 0."""
    def f(t):
        return struve_h._mp(0, t) / t

    return mp.quad(f, [0, x])


@reference_implementation(scipy=ufuncs.itairy)
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


@reference_implementation(scipy=ufuncs.itmodstruve0)
def itmodstruve0(x: Real) -> Real:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation(scipy=ufuncs.itstruve0)
def itstruve0(x: Real) -> Real:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation(scipy=ufuncs._iv_ratio)
def iv_ratio(v: Real, x: Real) -> Real:
    """Returns the ratio ``iv(v, x) / iv(v - 1, x)``"""
    numerator = cyl_bessel_i._mp(v, x)
    return numerator / cyl_bessel_i._mp(v - 1, x)


@reference_implementation(scipy=ufuncs._iv_ratio_c)
def iv_ratio_c(v: Real, x: Real) -> Real:
    """Returns ``1 - iv_ratio(v, x)``."""
    numerator = cyl_bessel_i._mp(v, x)
    denominator = cyl_bessel_i._mp(v - 1, x)
    # Set precision high enough to avoid catastrophic cancellation.
    # For large x, iv_ratio_c(v, x) ~ (v - 0.5) / x
    with mp.workprec(get_resolution_precision(x=(v - 0.5) / x)):
        result = mp.one - numerator / denominator
    return result


@reference_implementation(scipy=ufuncs.kei)
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


@reference_implementation(scipy=ufuncs.keip)
def keip(x: Real) -> Real:
    """Derivative of the Kelvin function kei."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.kelvin)
def kelvin(x: Real) -> Tuple[Complex, Complex, Complex, Complex]:
    """Kelvin functions as complex numbers."""
    be = mp.mpc(ber._mp(x), bei._mp(x))
    ke = mp.mpc(ker._mp(x), kei._mp(x))
    bep = mp.mpc(berp(to_fp(x)), beip(to_fp(x)))
    kep = mp.mpc(kerp(to_fp(x)), keip(to_fp(x)))
    return be, ke, bep, kep


@reference_implementation(scipy=ufuncs.ker)
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


@reference_implementation(scipy=ufuncs.kerp)
def kerp(x: Real) -> Real:
    """Derivative of the Kelvin function kerp."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs._kolmogc)
def kolmogc(x: Real) -> Real:
    """CDF of Kolmogorov distribution.

    CDF of Kolmogorov distribution can be expressed in terms of
    Jacobi Theta functions.
    TODO: Look into writing arbitrary precision reference implementations
    for kolmogc, kolmogci, kolmogi, and kolmogorov.
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs._kolmogci)
def kolmogci(x: Real) -> Real:
    """Inverse CDF of Kolmogorov distribution."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.kolmogi)
def kolmogi(x: Real) -> Real:
    """Inverse Survival Function of Kolmogorov distribution."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.kolmogorov)
def kolmogorov(x: Real) -> Real:
    """Survival Function of Kolmogorov distribution."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs._kolmogp)
def kolmogp(x: Real) -> Real:
    """Negative of PDF of Kolmogorov distribution.

    TODO: Why is this the negative pdf?
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs._lambertw)
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


@reference_implementation(scipy=ufuncs._lanczos_sum_expg_scaled)
def lanczos_sum_expg_scaled(z: Real) -> Real:
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    g = mp.mpf("6.024680040776729583740234375")
    return (mp.e / (z + g - 0.5)) ** (z - 0.5) * mp.gamma(z)


@reference_implementation(scipy=ufuncs._lgam1p)
def lgam1p(x: Real) -> Real:
    """Logarithm of abs(gamma(x + 1))."""
    if mp.isnan(x) or x == -mp.inf:
        return mp.nan
    if x == 0:
        return mp.zero
    if x == mp.inf:
        return mp.inf
    # set the precision high enough to resolve 1 + x.
    with mp.workprec(get_resolution_precision(x=x)):
        return gammaln._mp(x + 1)


@overload
def log1p(z: Real) -> Real: ...
@overload
def log1p(z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.log1p)
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


@reference_implementation(scipy=ufuncs._log1pmx)
def log1pmx(z):
    """log(z + 1) - z.

    Notes
    -----
    Branch cut on (-inf, -1)
    """
    # set the precision high enough to avoid catastrophic cancellation.
    # Near z = 0 log(1 + z) - z = -z^2/2 + O(z^3)
    with mp.workprec(get_resolution_precision(x=z/2)):
        result = log1p._mp(z) - z
    return result


@overload
def loggamma(z: Real) -> Real: ...
@overload
def loggamma(z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.loggamma)
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


@reference_implementation(scipy=ufuncs.log_expit)
def log_expit(x: Real) -> Real:
    """Log of `expit`."""
    return mp.log(mp.sigmoid(x))


@reference_implementation(scipy=ufuncs.log_wright_bessel, default_timeout=10)
def log_wright_bessel(a: Real, b: Real, x: Real) -> Real:
    """Natural logarithm of Wright's generalized Bessel function."""
    return mp.log(wright_bessel._mp(a, b, x))


@reference_implementation(scipy=ufuncs.logit)
def logit(p: Real) -> Real:
    """Logit function ``logit(p) = log(p/(1 - p))``"""
    if p == 1:
        return mp.inf
    if p < 0 or p > 1:
        return mp.nan
    # set the precision high enough to resolve 1 - p.
    with mp.workprec(get_resolution_precision(x=p)):
        result = mp.log(p/(1-p))
    return result


@reference_implementation(scipy=ufuncs.mathieu_modcem1)
def mcm1(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even modified Mathieu function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.mathieu_modcem2)
def mcm2(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Even modified Mathieu function of the second kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.modfresnelm)
def modified_fresnel_minus(x: Real) -> Tuple[Complex, Complex]:
    """Modified Fresnel negative integrals."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.modfresnelp)
def modified_fresnel_plus(x: Real) -> Tuple[Complex, Complex]:
    """Modified Fresnel negative integrals."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.mathieu_modsem1)
def msm1(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd modified Mathieu function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.mathieu_modsem2)
def msm2(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd modified Mathieu function of the second kind and its derivative."""
    raise NotImplementedError


@overload
def nbdtr(k: Integer, n: Integer, p: Real) -> Real: ...
@overload
def nbdtr(k: Real, n: Real, p: Real) -> Real: ...


@reference_implementation(scipy=ufuncs.nbdtr)
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


@reference_implementation(scipy=ufuncs.nbdtrc)
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


@reference_implementation(scipy=ufuncs.ndtr)
def ndtr(x):
    """Cumulative distribution of the standard normal distribution."""
    if x.imag == 0:
        return mp.ncdf(x.real)
    return (1 + erf._mp(x/mp.sqrt(2)))/2


@reference_implementation(scipy=ufuncs.ndtri)
def ndtri(y: Real) -> Real:
    """Inverse of `ndtr` vs x."""
    if not 0 <= y <= 1:
        return mp.nan
    if y == 0:
        return -mp.inf
    if y == 1:
        return mp.inf
    # set the precision high enough to resolve 2*y - 1
    with mp.workprec(get_resolution_precision(x=2*y)):
        result = mp.sqrt(2) * mp.erfinv(2*y - 1)
    return result


@reference_implementation(scipy=ufuncs.obl_ang1_cv)
def oblate_aswfa(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function obl_ang1 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_ang1)
def oblate_aswfa_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_rad1_cv)
def oblate_radial1(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function obl_rad1 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_rad1)
def oblate_radial1_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_rad2_cv)
def oblate_radial2(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal angular function obl_rad2 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_rad2)
def oblate_radial2_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Oblate spheroidal radial function of the second kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.obl_cv)
def oblate_segv(m: Real, n: Real, c: Real) -> Real:
    """Characteristic value of oblate spheroidal function."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.owens_t)
def owens_t(h: Real, a: Real) -> Real:
    """Owen's T Function."""
    def integrand(x):
        return mp.exp(-(h**2) * (1 + x**2) / 2) / (1 + x**2)

    return mp.quad(integrand, [0, a]) / (2 * mp.pi)


@reference_implementation(scipy=ufuncs.pbdv)
def pbdv(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function D."""
    d = mp.pcfd(v, x)
    _, dp = ufuncs.pbdv(to_fp(v), to_fp(x))
    message = (
        "Reference implementation pbdv falls back to SciPy for derivative."
    )
    warnings.warn(message, XSRefFallbackWarning)
    return d, to_mp(dp)


@reference_implementation(scipy=ufuncs.pbvv)
def pbvv(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function V."""
    # Set precision to guarantee -v - 0.5 retains precision for very small v.
    with mp.workprec(get_resolution_precision(x=2*v)):
        d = mp.pcfv(-v - 0.5, x)
    _, dp = ufuncs.pbvv(to_fp(v), to_fp(x))
    message = (
        "Reference implementation pbvv falls back to SciPy for derivative."
    )
    warnings.warn(message, XSRefFallbackWarning)
    return d, to_mp(dp)


@reference_implementation(scipy=ufuncs.pbwa)
def pbwa(v: Real, x: Real) -> Tuple[Real, Real]:
    """Parabolic cylinder function W."""
    d = mp.pcfw(v, x)
    _, dp = ufuncs.pbwa(to_fp(v), to_fp(x))
    message = (
        "Reference implementation pbwa falls back to SciPy for derivative."
    )
    warnings.warn(message, XSRefFallbackWarning)
    return d, to_mp(dp)


@reference_implementation(scipy=ufuncs.pdtr)
def pdtr(k: Real, m: Real) -> Real:
    """Poisson cumulative distribution function."""
    if k < 0 or m < 0:
        return mp.nan
    k = mp.floor(k)
    return gammaincc._mp(k + 1, m)


@reference_implementation(scipy=ufuncs.pdtrc)
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


@reference_implementation(scipy=ufuncs.pdtri)
def pdtri(k, y):
    """Inverse of `pdtr` vs m."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.lpmv)
def pmv(m: Integer, v: Real, x: Real) -> Real:
    """Associated Legendre function of integer order and real degree."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.poch)
def poch(z: Real, m: Real) -> Real:
    """Pochhammer symbol."""
    return mp.rf(z, m)


@reference_implementation(scipy=ufuncs.pro_ang1_cv)
def prolate_aswfa(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function pro_ang1 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_ang1)
def prolate_aswfa_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_rad1_cv)
def prolate_radial1(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function pro_rad1 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_rad1)
def prolate_radial1_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function of the first kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_rad2_cv)
def prolate_radial2(
    m: Real, n: Real, c: Real, cv: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal angular function pro_rad2 for precomputed cv

    cv: Characteristic Value
    """
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_rad2)
def prolate_radial2_nocv(
    m: Real, n: Real, c: Real, x: Real
) -> Tuple[Real, Real]:
    """Prolate spheroidal radial function of the second kind and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.pro_cv)
def prolate_segv(m: Real, n: Real, c: Real) -> Real:
    """Characteristic value of prolate spheroidal function."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.radian)
def radian(d: Real, m: Real, s: Real) -> Real:
    """Convert from degrees to radians."""
    return mp.radians(d + m / 60 + s / 3600)


@overload
def rgamma(z: Real) -> Real: ...
@overload
def rgamma(z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.rgamma)
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


@reference_implementation(scipy=ufuncs._riemann_zeta)
def riemann_zeta(z):
    """Riemann zeta function.

    Notes
    -----
    A single pole at z = 1
    """
    if z == 1.0:
        return mp.nan
    return mp.zeta(z)


@reference_implementation(scipy=ufuncs.round)
def round(x: Real) -> Real:
    """Round to the nearest integer."""
    return mp.nint(x)


@reference_implementation(scipy=ufuncs._scaled_exp1)
def scaled_exp1(x: Real) -> Real:
    """Exponentially scaled exponential integral E1."""
    return mp.exp(x) * x * mp.e1(x)


@reference_implementation(scipy=ufuncs.mathieu_sem)
def sem(m: Real, q: Real, x: Real) -> Tuple[Real, Real]:
    """Odd Mathieu function and its derivative."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.mathieu_b)
def sem_cva(m: Real, q: Real) -> Real:
    """Characteristic value of odd Mathieu functions."""
    raise NotImplementedError


@overload
def shichi(x: Real) -> Tuple[Real, Real]: ...
@overload
def shichi(x: Complex) -> Tuple[Complex, Complex]: ...


@reference_implementation(scipy=ufuncs.shichi)
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


@reference_implementation(scipy=ufuncs.sici)
def sici(x):
    """Sine and cosine integrals.

    Notes
    -----
    Sine and cosine integrals are entire functions
    """
    return mp.si(x), mp.ci(x)


@reference_implementation(scipy=ufuncs.sindg)
def sindg(x):
    """Sine of the angle `x` given in degrees."""
    raise NotImplementedError


@overload
def sinpi(x: Real) -> Real: ...
@overload
def sinpi(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs._sinpi)
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


@reference_implementation(scipy=ufuncs.smirnov)
def smirnov(n, d):
    """Kolmogorov-Smirnov complementary cumulative distribution function."""
    raise NotImplementedError


@overload
def smirnovc(n: Integer, p: Real) -> Real: ...
@overload
def smirnovc(n: Real, p: Real) -> Real: ...


@reference_implementation(scipy=ufuncs._smirnovc)
def smirnovc(n, d):
    """Kolmogorov-Smirnov cumulative distribution function."""
    raise NotImplementedError


@overload
def smirnovci(n: Integer, p: Real) -> Real: ...
@overload
def smirnovci(n: Real, p: Real) -> Real: ...


@reference_implementation(scipy=ufuncs._smirnovci)
def smirnovci(n, p):
    """Inverse to `smirnovc`."""
    raise NotImplementedError


@overload
def smirnovi(n: Integer, p: Real) -> Real: ...
@overload
def smirnovi(n: Real, p: Real) -> Real: ...


@reference_implementation(scipy=ufuncs.smirnovi)
def smirnovi(n, p):
    """Inverse to `smirnov`."""
    raise NotImplementedError


@overload
def smirnovp(n: Integer, d: Real) -> Real: ...
@overload
def smirnovp(n: Real, d: Real) -> Real: ...


@reference_implementation(scipy=ufuncs._smirnovp)
def smirnovp(n, d):
    """Negative of Kolmogorov-Smirnov pdf."""
    raise NotImplementedError


@overload
def spence(z: Real) -> Real: ...
@overload
def spence(z: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.spence)
def spence(z):
    """Spence's function, also known as the dilogarithm."""
    # set the precision high enough that mp.one - z != 1
    with mp.workprec(get_resolution_precision(x=z)):
        result = mp.polylog(2, mp.one - z)
    return result


@reference_implementation(scipy=ufuncs.struve)
def struve_h(v: Real, x: Real) -> Real:
    """Struve function."""
    return mp.struveh(v, x)


@reference_implementation(scipy=ufuncs.modstruve)
def struve_l(v: Real, x: Real) -> Real:
    """Modified Struve function."""
    return  mp.struvel(v, x)


@reference_implementation(scipy=ufuncs.tandg)
def tandg(x: Real) -> Real:
    """Tangent of angle x given in degrees."""
    raise NotImplementedError


@reference_implementation(scipy=ufuncs.voigt_profile)
def voigt_profile(x: Real, sigma: Real, gamma: Real) -> Real:
    """Voigt profile"""
    raise NotImplementedError


@overload
def wofz(x: Real) -> Real: ...
@overload
def wofz(x: Complex) -> Complex: ...


@reference_implementation(scipy=ufuncs.wofz)
def wofz(x):
    """Faddeeva function

    Notes
    -----
    wofz is an entire function
    """
    return mp.exp(-x**2) * mp.erfc(-mp.j * x)


@reference_implementation(scipy=ufuncs.wright_bessel, default_timeout=10)
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


@reference_implementation(scipy=ufuncs.xlogy)
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


@reference_implementation(scipy=ufuncs.xlog1py)
def xlog1py(x, y):
    """Compute ``x*log(y)`` so that the result is 0 if ``x = 0``."""
    if x == 0 and not (mp.isnan(x.real) or mp.isnan(x.imag)):
        return mp.zero
    return x * log1p._mp(y)


@reference_implementation(scipy=ufuncs._zeta)
def zeta(z: Real, q: Real) -> Real:
    """Hurwitz zeta function."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z, a=q)


@reference_implementation(scipy=ufuncs.zetac)
def zetac(z: Real) -> Real:
    """Riemann zeta function minus 1."""
    # set the precision high enough to avoid catastrophic cancellation.
    # As z approaches +inf in the right halfplane:
    # zeta(z) - 1 = 2^-z + O(3^-z).
    if z == 1:
        return mp.nan
    with mp.workprec(get_resolution_precision(log2abs_x=-z.real)):
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
    "get_resolution_precision",
    "is_complex",
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
