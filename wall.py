"""The square-root wall: exact formulas (Kummer and parabolic cylinder functions) and the log-time simulator.

Notation follows the note. B is a standard Brownian motion; under the clock s = log(1+t) the process
U_s = exp(-s/2) B_{exp(s)-1} is an Ornstein-Uhlenbeck process with generator L f = f''/2 - u f'/2, and the
moving walls c*sqrt(1+t) become the fixed levels +-c.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import erfi, hyp1f1, pbdv

SQ_HALF_PI = np.sqrt(np.pi / 2.0)
BGK_BETA = 0.5825971579390106          # -zeta(1/2)/sqrt(2*pi), the continuity-correction constant of [8]


# ── special functions ──────────────────────────────────────────────────────────────────────────────────────
def f_even(p, u):
    """M(-p, 1/2, u^2/2): the even solution of L f = -p f, equal to 1 at u = 0."""
    return hyp1f1(-p, 0.5, 0.5 * np.square(u))


def f_odd(p, u):
    """u M(1/2 - p, 3/2, u^2/2): the odd solution of L f = -p f, with slope 1 at u = 0."""
    return u * hyp1f1(0.5 - p, 1.5, 0.5 * np.square(u))


def h_plus(lam, u):
    """D_{2 lam}(-u), proportional to exp(-u^2/4) h(u) with h the solution of L h = -lam h of polynomial growth at -inf."""
    return pbdv(2.0 * lam, -u)[0]


def scale(u):
    """The OU scale function s(u) = int_0^u exp(v^2/2) dv."""
    return SQ_HALF_PI * erfi(np.asarray(u) / np.sqrt(2.0))


# ── roots: scan for the first sign change, then brentq (a fixed bracket fails) ─────────────────────────────
def first_root(g, lo, hi, n=4000, skip=0, xtol=1e-14):
    """The (skip+1)-th sign change of g on a scan of (lo, hi], refined by brentq."""
    xs = np.linspace(lo, hi, n + 1)[1:]
    vals = np.array([g(x) for x in xs])
    seen = 0
    for i in range(1, len(xs)):
        if vals[i - 1] == 0.0:
            if seen == skip:
                return float(xs[i - 1])
            seen += 1
            continue
        if np.sign(vals[i]) != np.sign(vals[i - 1]):
            if seen == skip:
                return float(brentq(g, xs[i - 1], xs[i], xtol=xtol, rtol=1e-15, maxiter=200))
            seen += 1
    raise ValueError(f"no sign change number {skip + 1} on ({lo}, {hi}]")


def lam_two(c, k=1):
    """lambda(c): the k-th zero in p > 0 of M(-p, 1/2, c^2/2) (k = 1: the tail exponent of the two-sided wall).
    k = 2 is the next even eigenvalue, the first one that also reaches the start u = 0."""
    guess = np.pi ** 2 / (8.0 * c * c)
    hi = (2 * k - 1) ** 2 * guess * 1.3 + 4.0 * k
    return first_root(lambda p: f_even(p, c), 0.0, hi, n=6000, skip=k - 1)


def lam_plus(c, k=1):
    """lambda^+(c): the k-th zero in lambda > 0 of D_{2 lambda}(-c) (k = 1: the one-sided wall's exponent)."""
    hi = 0.5 + 1.2 * k + 0.5
    return first_root(lambda l: h_plus(l, c), 0.0, hi, n=3000, skip=k - 1)


def c_threshold(p):
    """c_p: the first zero in c > 0 of M(-p, 1/2, c^2/2), the largest c with E[(1+tau_c)^p] finite."""
    return first_root(lambda c: f_even(p, c), 0.0, 6.0, n=6000)


def bracket_det(lam, a, b):
    """f_e(-a) f_o(b) - f_e(b) f_o(-a): zero exactly when some solution of L h = -lam h vanishes at -a and b."""
    return f_even(lam, a) * f_odd(lam, b) + f_even(lam, b) * f_odd(lam, a)


def lam_bracket(a, b, k=1):
    """lambda(a, b): the principal (k = 1) Dirichlet eigenvalue of L on (-a, b), the bracket's tail exponent."""
    hi = 2.0 * k * k * np.pi ** 2 / (2.0 * (a + b) ** 2) + 6.0 * k
    return first_root(lambda l: bracket_det(l, a, b), 0.0, hi, n=6000, skip=k - 1)


def bracket_mode(lam, a, b, u):
    """The Dirichlet eigenfunction on (-a, b) for eigenvalue lam: f_e(a) f_o(u) + f_o(a) f_e(u), zero at -a."""
    return f_even(lam, a) * f_odd(lam, u) + f_odd(lam, a) * f_even(lam, u)


def late_win_share(a, b, eps=1e-6):
    """The share of LATE exits that end at the target: the principal mode's exit flux |phi'| e^{-u^2/2} at b,
    over the sum of the fluxes at b and -a (the killed density decays as e^{-lam s} phi(u) e^{-u^2/2})."""
    lam = lam_bracket(a, b)
    d = lambda u: (bracket_mode(lam, a, b, u + eps) - bracket_mode(lam, a, b, u - eps)) / (2 * eps)
    fb, fa = abs(d(b)) * np.exp(-b * b / 2), abs(d(-a)) * np.exp(-a * a / 2)
    return float(fb / (fb + fa))


def win(a, b):
    """P(the widening bracket ends at the target +b before the stop -a) = int_{-a}^0 e^{v^2/2} / int_{-a}^b e^{v^2/2}."""
    return float(scale(a) / (scale(a) + scale(b)))


def moment(p, c):
    """E[(1+tau_c)^p] = 1/M(-p, 1/2, c^2/2) for p < lambda(c), inf otherwise."""
    return float(1.0 / f_even(p, c)) if p < lam_two(c) else np.inf


def mean_tau(c):
    return c * c / (1.0 - c * c) if c < 1 else np.inf


def second_moment(c):
    """E[(1+tau_c)^2] = 3 / (3 - 6 c^2 + c^4), finite for c^2 < 3 - sqrt(6)."""
    d = 3.0 - 6.0 * c ** 2 + c ** 4
    return 3.0 / d if c * c < 3.0 - np.sqrt(6.0) else np.inf


# ── an independent route: the Schrodinger form of L, discretised ───────────────────────────────────────────
def lam_fd(lo, hi, n=6000):
    """Principal Dirichlet eigenvalue of L on (lo, hi) by second-order finite differences.

    h = exp(u^2/4) phi turns L h = -lam h into -phi''/2 + (u^2/8 - 1/4) phi = lam phi, a symmetric problem.
    No special function enters, so it checks the Kummer and parabolic-cylinder roots independently."""
    from scipy.linalg import eigh_tridiagonal
    u = np.linspace(lo, hi, n + 2)[1:-1]
    h = u[1] - u[0]
    d = 1.0 / h ** 2 + u ** 2 / 8.0 - 0.25
    e = np.full(n - 1, -0.5 / h ** 2)
    w = eigh_tridiagonal(d, e, select="i", select_range=(0, 0), eigvals_only=True)
    return float(w[0])


# ── the simulator: exact OU steps in log time, bridge-corrected crossing test ──────────────────────────────
def exit_log_time(n, upper, lower=None, dt=1e-3, s_max=60.0, seed=0, bridge=True, shift=0.0):
    """First exit of U from (lower, upper) in log time, for n independent paths started at 0.

    upper, lower: the walls in units of sqrt(1+t) (lower=None: a one-sided wall). shift > 0 moves both walls
    outward by shift*sqrt(1+t)^{-1} in U, i.e. by the constant `shift` in B (the wall c sqrt(1+t) + shift).
    Each step is the exact OU transition U' = e^{-dt/2} U + sqrt(1 - e^{-dt}) xi. A path stops if U' is outside;
    if both ends are inside it stops with the Brownian-bridge probability exp(-2 d d' / dt) for each wall, d and
    d' the distances to that wall at the two ends. The exit is recorded at the step's midpoint.
    Returns S (log time, s_max where censored) and side (+1 upper, -1 lower, 0 censored)."""
    rng = np.random.default_rng(seed)
    U = np.zeros(n)
    idx = np.arange(n)
    S = np.full(n, float(s_max))
    side = np.zeros(n, dtype=np.int8)
    k_ou, sd = np.exp(-dt / 2.0), np.sqrt(-np.expm1(-dt))
    s = 0.0
    while idx.size and s < s_max - 1e-12:
        m = idx.size
        Un = k_ou * U + sd * rng.standard_normal(m)
        sn = s + dt
        up0, up1 = upper + shift * np.exp(-s / 2.0), upper + shift * np.exp(-sn / 2.0)
        hit_up = Un >= up1
        if lower is None:
            hit_lo = np.zeros(m, dtype=bool)
        else:
            lo0, lo1 = lower - shift * np.exp(-s / 2.0), lower - shift * np.exp(-sn / 2.0)
            hit_lo = Un <= lo1
        if bridge:
            inside = ~(hit_up | hit_lo)
            du = np.where(inside, (up0 - U) * (up1 - Un), 1.0)
            bu = inside & (rng.random(m) < np.exp(-2.0 * du / dt))
            if lower is not None:
                dl = np.where(inside, (U - lo0) * (Un - lo1), 1.0)
                pl = np.exp(-2.0 * dl / dt)
                bl = inside & (rng.random(m) < pl)
                both = bu & bl
                if both.any():                      # both walls in one step: split by their bridge weights
                    pu = np.exp(-2.0 * du[both] / dt)
                    go_up = rng.random(both.sum()) * (pu + pl[both]) < pu
                    bu[both], bl[both] = go_up, ~go_up
                hit_lo = hit_lo | bl
            hit_up = hit_up | bu
        done = hit_up | hit_lo
        if done.any():
            j = idx[done]
            S[j] = s + 0.5 * dt
            side[j] = np.where(hit_up[done], 1, -1)
            keep = ~done
            idx, U = idx[keep], Un[keep]
        else:
            U = Un
        s = sn
    return S, side


def daily_check(n, c, dt, t_max, seed=0, chunk=1_000_000):
    """The wall checked only on the grid t = dt, 2 dt, ...: the first grid time with |B_t| >= c sqrt(1+t).
    Exact Gaussian increments; no bridge correction (that is the point). Returns tau (t_max+dt where censored)."""
    out = []
    rng = np.random.default_rng(seed)
    sq = np.sqrt(dt)
    kmax = int(round(t_max / dt))
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        B = np.zeros(m)
        idx = np.arange(m)
        tau = np.full(m, (kmax + 1) * dt)
        k = 0
        while idx.size and k < kmax:
            k += 1
            B = B + sq * rng.standard_normal(idx.size)
            hit = np.abs(B) >= c * np.sqrt(1.0 + k * dt)
            if hit.any():
                tau[idx[hit]] = k * dt
                idx, B = idx[~hit], B[~hit]
        out.append(tau)
    return np.concatenate(out)


def clock_paths(n_paths, t_end, dt, c, seed):
    """Brownian paths on a fine grid in t, with the first grid time each reaches |B| >= c sqrt(1+t) (nan if none)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + dt / 2, dt)
    inc = np.sqrt(dt) * rng.standard_normal((n_paths, t.size - 1))
    B = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    hit = np.abs(B) >= c * np.sqrt(1.0 + t)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return t, B, first
