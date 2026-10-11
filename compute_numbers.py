"""Every number and array in the square-root-wall note, computed once.

Writes results.npz (the arrays the three figures read), numbers.json (every number the prose states) and
CHECKS.txt (each check, its tolerance and PASS/FAIL), all beside this script. Run: python3 compute_numbers.py
Seeds are fixed and recorded in numbers.json; the whole run takes a few minutes on one core."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import wall as W

HERE = Path(__file__).resolve().parent
T0 = time.time()
OUT = {"seeds": {}, "runtime_s": {}}
CHECKS = []
INFO = []          # reported beside the checks, never scored: estimators whose variance is infinite


def check(name, value, target, tol, ok=None, note=""):
    """Record one check: |value - target| <= tol unless an explicit verdict is given."""
    if ok is None:
        ok = bool(abs(value - target) <= tol)
    CHECKS.append((name, value, target, tol, "PASS" if ok else "FAIL", note))
    return ok


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def stamp(key):
    OUT["runtime_s"][key] = round(time.time() - T0, 1)
    print(f"  [{key}] {OUT['runtime_s'][key]:.0f}s", flush=True)


# ── 1. exact: thresholds, exponents, closed forms, brackets ─────────────────────────────────────────────────
print("exact quantities ...", flush=True)
P_LIST = [0.5, 1, 2, 3, 4]
c_p = {str(p): W.c_threshold(p) for p in P_LIST}
OUT["c_p"] = c_p
OUT["c2_closed"] = float(np.sqrt(3 - np.sqrt(6)))
check("c_1 = 1 (first zero of 1 - c^2)", c_p["1"], 1.0, 1e-9)
check("c_2 = sqrt(3 - sqrt 6)", c_p["2"], OUT["c2_closed"], 1e-9)
for p in P_LIST:                    # the same zeros found the other way: lambda(c_p) = p, a root in p, not in c
    check(f"lambda(c_p) = p at p = {p}", W.lam_two(c_p[str(p)]), float(p), 1e-8)

C4 = [0.5, 1.0, 1.5, 2.0]
ex = {}
for c in C4:
    lt, lt2 = W.lam_two(c), W.lam_two(c, 2)
    lp, lp2 = W.lam_plus(c), W.lam_plus(c, 2)
    fd_t, fd_p = W.lam_fd(-c, c), W.lam_fd(-12.0, c)
    ex[str(c)] = dict(lam=lt, lam_next_even=lt2, lam_plus=lp, lam_plus_next=lp2, fd_lam=fd_t, fd_lam_plus=fd_p)
    check(f"lambda({c}): Kummer root vs finite-difference eigenvalue (rel)", abs(lt - fd_t) / lt, 0, 1e-5)
    check(f"lambda+({c}): parabolic-cylinder root vs finite-difference eigenvalue on (-12, c) (rel)",
          abs(lp - fd_p) / lp, 0, 1e-5)
OUT["exponents"] = ex
check("lambda(1) = 1", W.lam_two(1.0), 1.0, 1e-9)

small = {str(c): W.lam_two(c) * 8 * c * c / np.pi ** 2 for c in (0.05, 0.02, 0.01)}
OUT["small_c_ratio"] = small
r = [small[k] for k in ("0.05", "0.02", "0.01")]
check("lambda(c) 8c^2/pi^2 -> 1 as c -> 0 (at c = 0.01; monotone in c)", r[2], 1.0, 1e-4,
      ok=bool(abs(r[2] - 1) < 1e-4 and abs(r[0] - 1) > abs(r[1] - 1) > abs(r[2] - 1)))
OUT["lam_plus_at_1e-3"] = W.lam_plus(1e-3)
check("lambda+(c) -> 1/2 as c -> 0 (c = 0.001)", OUT["lam_plus_at_1e-3"], 0.5, 1e-3)
OUT["lam_at_3"], OUT["lam_plus_at_3"] = W.lam_two(3.0), W.lam_plus(3.0)

grid = np.linspace(0.05, 3.0, 320)
lam_g = np.array([W.lam_two(c) for c in grid])
lamp_g = np.array([W.lam_plus(c) for c in grid])
check("lambda strictly decreasing on the c grid (320 points)", float(np.max(np.diff(lam_g))), 0, 0,
      ok=bool(np.all(np.diff(lam_g) < 0)))
check("lambda+ strictly decreasing on the c grid", float(np.max(np.diff(lamp_g))), 0, 0,
      ok=bool(np.all(np.diff(lamp_g) < 0)))
check("0 < lambda+ < 1/2 on the c grid", float(lamp_g.max()), 0.5, 0,
      ok=bool(np.all((lamp_g > 0) & (lamp_g < 0.5))))
stamp("exponent grid")

MOM_C = [0.3, 0.5, 0.7, 0.8, 0.9]
closed = {}
for c in MOM_C:
    m1, m2 = W.mean_tau(c), W.second_moment(c)
    closed[str(c)] = dict(lam=W.lam_two(c), mean=m1, second=m2,
                          var=(m2 - (1 + m1) ** 2) if np.isfinite(m2) else None,
                          half=float(1 / W.f_even(0.5, c)))
    check(f"Kummer polynomial: 1/M(-1, 1/2, c^2/2) = 1/(1-c^2) at c={c}", 1 / W.f_even(1, c), 1 / (1 - c * c), 1e-12)
    check(f"Kummer polynomial: 1/M(-2, 1/2, c^2/2) = 3/(3-6c^2+c^4) at c={c}", 1 / W.f_even(2, c), 3 / (3 - 6 * c * c + c ** 4), 1e-11)
OUT["closed"] = closed

BRK = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (0.5, 1.0), (2.0, 1.0)]
brk = {}
for a, b in BRK:
    w = W.win(a, b)
    wq = quad(lambda v: np.exp(v * v / 2), -a, 0)[0] / quad(lambda v: np.exp(v * v / 2), -a, b)[0]
    la, la2 = W.lam_bracket(a, b), W.lam_bracket(a, b, 2)
    brk[f"{a:g},{b:g}"] = dict(a=a, b=b, win=w, fixed=a / (a + b), ratio_fixed_over_win=(a / (a + b)) / w,
                               lam=la, lam_next=la2, fd_lam=W.lam_fd(-a, b), late_win_share=W.late_win_share(a, b))
    check(f"win({a:g},{b:g}): erfi form vs quadrature", w, wq, 1e-12)
    check(f"lambda({a:g},{b:g}): Kummer determinant root vs finite differences (rel)",
          abs(la - brk[f"{a:g},{b:g}"]["fd_lam"]) / la, 0, 1e-5)
OUT["brackets"] = brk
check("win(1,1) = 1/2", brk["1,1"]["win"], 0.5, 1e-14)
check("win(1,2) + win(2,1) = 1", brk["1,2"]["win"] + brk["2,1"]["win"], 1.0, 1e-14)
check("lambda(1,1) = lambda(1) = 1", brk["1,1"]["lam"], 1.0, 1e-9)

OUT["scale_g"] = {"1": float(W.scale(1.0)), "2": float(W.scale(2.0)), "0.5": float(W.scale(0.5)),
                  "3": float(W.scale(3.0))}
OUT["fd_max_rel"] = {"two": max(abs(e["lam"] - e["fd_lam"]) / e["lam"] for e in ex.values()),
                     "one": max(abs(e["lam_plus"] - e["fd_lam_plus"]) / e["lam_plus"] for e in ex.values()),
                     "bracket": max(abs(v["lam"] - v["fd_lam"]) / v["lam"] for v in brk.values())}
OUT["fig_reads"] = {"win_1_4": W.win(1, 4), "win_1_05": W.win(1, 0.5), "win_05_4": W.win(0.5, 4),
                    "win_2_4": W.win(2, 4), "win_2_2": W.win(2, 2), "lam_plus_grid_max": float(lamp_g.max()),
                    "lam_grid_at_005": float(lam_g[0]), "c_where_lam_is_5": W.first_root(
                        lambda c: W.lam_two(c) - 5.0, 0.3, 0.7, n=40)}
bgrid = np.linspace(0.25, 4.0, 301)
wins = {a: np.array([W.win(a, b) for b in bgrid]) for a in (0.5, 1.0, 2.0)}
lower = all(np.all(wins[a][bgrid > a + 1e-9] < (a / (a + bgrid))[bgrid > a + 1e-9]) for a in wins)
check("widening bracket wins less than a/(a+b) whenever b > a (three stops, 301 targets)", 0, 0, 0, ok=bool(lower))
stamp("exact")

# ── 2. Monte Carlo: mean and second moment ─────────────────────────────────────────────────────────────────
print("moments ...", flush=True)
N_MOM, S_MAX = 1_000_000, 60.0
mom = {}
for i, c in enumerate([0.3, 0.5, 0.7, 0.9]):
    seed = 20250908 + i
    dt = c * c / 100
    S, side = W.exit_log_time(N_MOM, c, -c, dt=dt, s_max=S_MAX, seed=seed)
    tau = np.expm1(S)
    q = (1 + tau) ** 2
    m1, m2 = W.mean_tau(c), W.second_moment(c)
    rec = dict(c=c, n=N_MOM, dt=dt, s_max=S_MAX, seed=seed, censored=float(np.mean(side == 0)),
               mean=float(tau.mean()), mean_se=se(tau), mean_exact=m1,
               second=float(q.mean()), second_se=se(q), second_exact=m2,
               upper_share=float(np.mean(side == 1)), max_tau=float(tau.max()))
    rec["mean_dev_se"] = (rec["mean"] - m1) / rec["mean_se"]
    rec["mean_rel"] = (rec["mean"] - m1) / m1
    rec["second_dev_se"] = (rec["second"] - m2) / rec["second_se"]
    rec["second_rel"] = (rec["second"] - m2) / m2
    # batch spread: twenty independent batches, the honest error bar when the variance is infinite
    bm = tau.reshape(20, -1).mean(axis=1)
    rec["mean_batch_sd"] = float(bm.std(ddof=1))
    rec["mean_batch_min"], rec["mean_batch_max"] = float(bm.min()), float(bm.max())
    if c == 0.9:                     # a moment whose estimator HAS a finite variance: p = 1/2 < lambda(0.9)/2
        h = np.sqrt(1 + tau)
        rec.update(half=float(h.mean()), half_se=se(h), half_exact=float(1 / W.f_even(0.5, c)))
        rec["half_dev_se"] = (rec["half"] - rec["half_exact"]) / rec["half_se"]
    mom[str(c)] = rec
    OUT["seeds"][f"moments_c{c}"] = seed
    print(f"  c={c}: E tau {rec['mean']:.5f} +- {rec['mean_se']:.5f} vs {m1:.5f}; "
          f"E(1+tau)^2 {rec['second']:.5f} +- {rec['second_se']:.5f} vs {m2:.5f}", flush=True)
OUT["moments"] = mom
for c in ("0.3", "0.5", "0.7"):
    check(f"MC mean, c={c}, within 3 SE of c^2/(1-c^2)", mom[c]["mean_dev_se"], 0, 3)
INFO.append(f"MC mean at c=0.9: {mom['0.9']['mean']:.5f}, naive SE {mom['0.9']['mean_se']:.5f}, against "
            f"{mom['0.9']['mean_exact']:.5f} ({100 * mom['0.9']['mean_rel']:+.2f}%); tau has infinite variance at "
            f"c=0.9 (lambda = {closed['0.9']['lam']:.4f} < 2), so the SE is not a valid error bar; twenty batch means "
            f"range {mom['0.9']['mean_batch_min']:.3f} to {mom['0.9']['mean_batch_max']:.3f}")
check("MC E[(1+tau)^(1/2)], c=0.9 (finite variance), within 3 SE of 1/M(-1/2, 1/2, c^2/2)",
      mom["0.9"]["half_dev_se"], 0, 3)
for c in ("0.3", "0.5"):
    check(f"MC second moment, c={c}, within 3 SE of 3/(3-6c^2+c^4)", mom[c]["second_dev_se"], 0, 3)
INFO.append(f"MC second moment at c=0.7: {mom['0.7']['second']:.4f}, naive SE {mom['0.7']['second_se']:.4f}, "
            f"against {mom['0.7']['second_exact']:.4f} ({100 * mom['0.7']['second_rel']:+.1f}%); (1+tau)^2 has infinite "
            f"variance at c=0.7 (lambda = {closed['0.7']['lam']:.4f} < 4), so the SE is not a valid error bar")
# beyond c = 1: the mean is infinite, a fractional moment is not (and its estimator has a finite variance)
seed = 20250908 + 9
S, side = W.exit_log_time(N_MOM, 1.2, -1.2, dt=1.44 / 100, s_max=S_MAX, seed=seed)
q = np.exp(S / 4.0)
mom["1.2"] = dict(c=1.2, n=N_MOM, dt=0.0144, seed=seed, censored=float(np.mean(side == 0)), lam=W.lam_two(1.2),
                  quarter=float(q.mean()), quarter_se=se(q), quarter_exact=float(1 / W.f_even(0.25, 1.2)))
mom["1.2"]["quarter_dev_se"] = (mom["1.2"]["quarter"] - mom["1.2"]["quarter_exact"]) / mom["1.2"]["quarter_se"]
OUT["seeds"]["moments_c1.2"] = seed
check("MC E[(1+tau)^(1/4)], c=1.2 (mean infinite; this moment finite, finite variance), within 3 SE",
      mom["1.2"]["quarter_dev_se"], 0, 3)
check("two-sided exits split evenly (c=0.5)", mom["0.5"]["upper_share"], 0.5, 3 * 0.5 / np.sqrt(N_MOM))
stamp("moments")

# ── 3. the bridge correction: first-order bias without it, none with it ────────────────────────────────────
print("bridge correction ...", flush=True)
bias = []
for dt in (0.01, 0.0025):
    for br in (False, True):
        S, _ = W.exit_log_time(N_MOM, 0.5, -0.5, dt=dt, s_max=S_MAX, seed=777, bridge=br)
        tau = np.expm1(S)
        bias.append(dict(dt=dt, bridge=br, mean=float(tau.mean()), se=se(tau),
                         bias=float(tau.mean() - W.mean_tau(0.5)),
                         dev_se=float((tau.mean() - W.mean_tau(0.5)) / se(tau))))
OUT["bridge_bias"] = bias
OUT["seeds"]["bridge_bias"] = 777
nb = [x for x in bias if not x["bridge"]]
yb = [x for x in bias if x["bridge"]]
check("uncorrected bias is O(sqrt dt): ratio at dt and dt/4 near 2", nb[0]["bias"] / nb[1]["bias"], 2.0, 0.3)
check("uncorrected bias exceeds 10 SE at both steps", min(x["dev_se"] for x in nb), 10, 0,
      ok=bool(min(x["dev_se"] for x in nb) > 10))
check("bridge-corrected mean within 3 SE at both steps", max(abs(x["dev_se"]) for x in yb), 0, 3)
stamp("bridge")


# ── 4. tail slopes in log time ─────────────────────────────────────────────────────────────────────────────
def ls_slope(S, s_pts, nb=20):
    """Least-squares slope of log P(S > s) on the points s_pts; standard error from nb independent batches."""
    def one(x):
        surv = np.array([(x > s).sum() for s in s_pts], dtype=float)
        return -np.polyfit(s_pts, np.log(np.maximum(surv, 0.5) / x.size), 1)[0]
    whole = one(S)
    parts = np.array([one(x) for x in np.array_split(S, nb)])
    return float(whole), float(parts.std(ddof=1) / np.sqrt(nb))


print("tails ...", flush=True)
N_TAIL = 1_000_000
tails = []
for j, (wall, c) in enumerate([(w, c) for w in ("two", "one") for c in C4]):
    e = ex[str(c)]
    lam, lam2 = (e["lam"], e["lam_next_even"]) if wall == "two" else (e["lam_plus"], e["lam_plus_next"])
    s1 = np.log(1000.0) / (lam2 - lam)                  # the next mode has fallen by a factor 1000
    s2 = min(np.log(N_TAIL / 2000.0) / lam, 80.0)       # about 2000 survivors expected at the end
    dt = 0.0025 if (wall == "two" and c == 0.5) else 0.01
    seed = 31000 + j
    S, side = W.exit_log_time(N_TAIL, c, (-c if wall == "two" else None), dt=dt, s_max=s2 + 2 * dt, seed=seed)
    pts = np.linspace(s1, s2, 50)
    sl, sl_se = ls_slope(S, pts)
    rec = dict(wall=wall, c=c, lam=lam, lam_next=lam2, s1=float(s1), s2=float(s2), t1=float(np.expm1(s1)),
               t2=float(np.expm1(s2)), dt=dt, n=N_TAIL, seed=seed, slope=sl, slope_se=sl_se,
               dev_se=(sl - lam) / sl_se, rel=(sl - lam) / lam, alive_s1=int((S > s1).sum()),
               alive_s2=int((S > s2).sum()))
    tails.append(rec)
    OUT["seeds"][f"tail_{wall}_c{c}"] = seed
    print(f"  {wall} c={c}: slope {sl:.4f} +- {sl_se:.4f} vs {lam:.4f} on s in [{s1:.2f}, {s2:.2f}]", flush=True)
    check(f"tail slope, {wall}-sided c={c}, within 3 SE of the exponent", rec["dev_se"], 0, 3)
OUT["tails"] = tails
stamp("tails")

# ── 5. brackets ────────────────────────────────────────────────────────────────────────────────────────────
print("brackets ...", flush=True)
mcb = {}
for j, (a, b) in enumerate(BRK):
    seed = 41000 + j
    dt = min(a, b) ** 2 / 100
    S, side = W.exit_log_time(N_MOM, b, -a, dt=dt, s_max=80.0, seed=seed)
    k = f"{a:g},{b:g}"
    wv = (side == 1).astype(float)
    rec = dict(a=a, b=b, n=N_MOM, dt=dt, seed=seed, censored=float(np.mean(side == 0)),
               win=float(wv.mean()), win_se=se(wv), exact=brk[k]["win"], fixed=brk[k]["fixed"])
    rec["dev_se"] = (rec["win"] - rec["exact"]) / rec["win_se"]
    rec["fixed_dev_se"] = (rec["win"] - rec["fixed"]) / rec["win_se"]
    # the outcome in units of the per-unit-time volatility: +b sqrt(1+tau) at the target, -a sqrt(1+tau) at the stop
    pnl = np.where(side == 1, b, -a) * np.exp(S / 2.0) * (side != 0)
    bm = pnl.reshape(20, -1).mean(axis=1)
    absp = np.abs(pnl)
    late = S > 5.0
    rec.update(pnl_mean=float(pnl.mean()), pnl_se_naive=se(pnl), pnl_batch_min=float(bm.min()),
               pnl_batch_max=float(bm.max()), pnl_batch_sd=float(bm.std(ddof=1)),
               pnl_max_share=float(absp.max() / absp.sum()), late_n=int(late.sum()),
               late_win=float(np.mean(side[late] == 1)) if late.any() else None,
               late_win_se=float(np.sqrt(np.mean(side[late] == 1) * (1 - np.mean(side[late] == 1)) / late.sum()))
               if late.any() else None, late_win_exact=brk[k]["late_win_share"])
    mcb[k] = rec
    OUT["seeds"][f"bracket_{k}"] = seed
    print(f"  ({k}): win {rec['win']:.5f} +- {rec['win_se']:.5f} vs {rec['exact']:.5f} (fixed {rec['fixed']:.4f});"
          f" late share {rec['late_win']} vs {rec['late_win_exact']:.4f}", flush=True)
    check(f"bracket ({k}) win rate within 3 SE of the scale-function formula", rec["dev_se"], 0, 3)
OUT["mc_brackets"] = mcb
check("2:1 bracket's late exits split as the principal eigenfunction's flux predicts (3 SE)",
      (mcb["1,2"]["late_win"] - mcb["1,2"]["late_win_exact"]) / mcb["1,2"]["late_win_se"], 0, 3)
stamp("brackets")

# ── 6. the once-a-day check ────────────────────────────────────────────────────────────────────────────────
print("daily checks ...", flush=True)


def predicted_log_survival(c, delta, s):
    """Adiabatic prediction: a check every delta in t acts as the wall c sqrt(1+t) + beta sqrt(delta) [8], i.e.
    the level c + beta sqrt(delta) e^{-s/2} in log time, with local exponent lambda of that level."""
    shift = W.BGK_BETA * np.sqrt(delta)
    cs = np.linspace(c, c + shift, 81)
    lam_c = np.array([W.lam_two(x) for x in cs])
    r = np.linspace(0, s.max(), 4001)
    lam_r = np.interp(c + shift * np.exp(-r / 2), cs, lam_c)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (lam_r[1:] + lam_r[:-1]) * np.diff(r))])
    return -np.interp(s, r, cum)


daily = []
for j, (c, delta) in enumerate([(0.5, 1.0), (0.5, 0.1), (0.8, 1.0), (0.8, 0.1)]):
    n = 4_000_000 if c == 0.5 else 2_000_000
    t_max = 1000.0 if c == 0.5 else (200000.0 if delta == 1.0 else 20000.0)
    seed = 51000 + j
    tau = W.daily_check(n, c, delta, t_max, seed=seed)
    cens = tau > t_max
    S = np.log1p(tau)
    # window: from where the survival first falls below 10 per cent to where fewer than 1000 paths remain
    grid_t = delta * np.arange(1, int(round(t_max / delta)) + 1)
    surv = n - np.searchsorted(np.sort(tau), grid_t, side="right")
    i1 = int(np.argmax(surv <= 0.1 * n))
    i2 = int(np.argmax(surv < 1000)) - 1
    pts_t = grid_t[i1:i2 + 1]
    if pts_t.size > 400:
        pts_t = pts_t[np.unique(np.linspace(0, pts_t.size - 1, 400).astype(int))]
    pts = np.log1p(pts_t)
    sl, sl_se = ls_slope(S, pts)
    lp = predicted_log_survival(c, delta, pts)
    pred = -np.polyfit(pts, lp, 1)[0]
    half = pts.size // 2
    halves = [ls_slope(S, pts[:half]), ls_slope(S, pts[half:])]
    pred_h = [-np.polyfit(pts[:half], lp[:half], 1)[0], -np.polyfit(pts[half:], lp[half:], 1)[0]]
    shift = W.BGK_BETA * np.sqrt(delta)
    S_sh, side_sh = W.exit_log_time(1_000_000, c, -c, dt=c * c / 100, s_max=60.0, seed=seed + 500, shift=shift)
    tau_sh = np.expm1(S_sh)
    m_exact = W.mean_tau(c)
    rec = dict(c=c, delta=delta, n=n, t_max=t_max, seed=seed, censored=float(cens.mean()),
               mean=float(tau.mean()), mean_se=se(tau), mean_exact=m_exact,
               shifted_mean=float(tau_sh.mean()), shifted_mean_se=se(tau_sh), shift=float(shift),
               shifted_seed=seed + 500,
               t1=float(pts_t[0]), t2=float(pts_t[-1]), n_pts=int(pts.size), slope=sl, slope_se=sl_se,
               lam=W.lam_two(c), slope_predicted=float(pred), t_mid=float(pts_t[half]),
               slope_early=halves[0][0], slope_early_se=halves[0][1], slope_late=halves[1][0],
               slope_late_se=halves[1][1], pred_early=float(pred_h[0]), pred_late=float(pred_h[1]),
               lam_at_t1=W.lam_two(c + shift / np.sqrt(1 + pts_t[0])),
               lam_at_t2=W.lam_two(c + shift / np.sqrt(1 + pts_t[-1])))
    rec["share_explained"] = (rec["shifted_mean"] - m_exact) / (rec["mean"] - m_exact)
    # the mean that the censored paths would add, bounded with the exact exponent beyond t_max
    pT = float(cens.mean())
    rec["censor_bound"] = pT * ((1 + t_max) / (rec["lam"] - 1)) if rec["lam"] > 1 else None
    rec["mean_ratio"] = rec["mean"] / m_exact
    rec["slope_over_lam"] = sl / rec["lam"]
    rec["slope_vs_pred_rel"] = (sl - pred) / pred
    rec["n_censored"] = int(cens.sum())
    daily.append(rec)
    OUT["seeds"][f"daily_c{c}_d{delta}"] = seed
    print(f"  c={c} delta={delta}: mean {rec['mean']:.4f} +- {rec['mean_se']:.4f} (continuous {m_exact:.4f}, "
          f"shifted wall {rec['shifted_mean']:.4f}); slope {sl:.3f} +- {sl_se:.3f} on t in "
          f"[{rec['t1']:.1f}, {rec['t2']:.1f}], predicted {pred:.3f}, lambda {rec['lam']:.3f}", flush=True)
    check(f"daily check c={c} dt={delta}: mean wait longer than the continuous wall's", rec["mean"], m_exact, 0,
          ok=bool(rec["mean"] - 3 * rec["mean_se"] > m_exact))
    check(f"daily check c={c} dt={delta}: tail slope within 3 SE + 5% of the shifted-wall prediction",
          sl - pred, 0, 3 * sl_se + 0.05 * pred)
OUT["daily"] = daily
stamp("daily")

scored = [c for c in CHECKS if "SE" in c[0] and "within 3 SE" in c[0]]
OUT["max_abs_dev_se_scored_mc"] = max(abs(c[1]) for c in scored)
OUT["n_scored_mc"] = len(scored)

# ── 7. the paths for the clock figure ──────────────────────────────────────────────────────────────────────
print("clock paths ...", flush=True)
t, B, first = W.clock_paths(400, 30.0, 0.005, 1.0, seed=2025)
OUT["seeds"]["clock_paths"] = 2025
s_stop = np.where(first >= 0, np.log1p(t[np.maximum(first, 0)]), np.inf)
targets = [0.25, 0.7, 1.3, 2.0, 2.8]
pick = []
for g in targets:
    cand = [i for i in np.argsort(np.abs(s_stop - g)) if i not in pick]
    pick.append(int(cand[0]))
pick.append(int(np.flatnonzero(first < 0)[0]))            # one path still inside at t = 30
OUT["clock"] = dict(paths=pick, stop_t=[float(t[first[i]]) if first[i] >= 0 else None for i in pick],
                    stop_s=[float(np.log1p(t[first[i]])) if first[i] >= 0 else None for i in pick],
                    share_unstopped_by_30=float(np.mean(first < 0)), pool=400, dt=0.005,
                    p_unstopped_exact_note="P(1+tau > 31) for c = 1 decays as 31^(-1)")
stamp("clock")

# ── 8. write ───────────────────────────────────────────────────────────────────────────────────────────────
np.savez_compressed(HERE / "results.npz", c_grid=grid, lam_grid=lam_g, lam_plus_grid=lamp_g,
                    b_grid=bgrid, win_a05=wins[0.5], win_a1=wins[1.0], win_a2=wins[2.0],
                    mc_a=np.array([mcb[f"{a:g},{b:g}"]["a"] for a, b in BRK]),
                    mc_b=np.array([mcb[f"{a:g},{b:g}"]["b"] for a, b in BRK]),
                    mc_win=np.array([mcb[f"{a:g},{b:g}"]["win"] for a, b in BRK]),
                    mc_win_se=np.array([mcb[f"{a:g},{b:g}"]["win_se"] for a, b in BRK]),
                    c_p=np.array([c_p[str(p)] for p in P_LIST]), p_list=np.array(P_LIST, dtype=float),
                    clock_t=t, clock_B=B[pick], clock_first=first[pick])
OUT["runtime_s"]["total"] = round(time.time() - T0, 1)
(HERE / "numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
lines = [f"CHECKS for the square-root-wall note — {len(CHECKS)} checks, "
         f"{sum(c[4] == 'PASS' for c in CHECKS)} PASS, {sum(c[4] == 'FAIL' for c in CHECKS)} FAIL", ""]
for name, v, tgt, tol, verdict, note in CHECKS:
    lines.append(f"{verdict}  {name}: value {v:.10g}, target {tgt:.10g}, tolerance {tol:.3g}" + (f"  ({note})" if note else ""))
lines += ["", "Reported, not scored (an estimator with infinite variance has no valid standard error):"]
lines += [f"INFO  {x}" for x in INFO]
(HERE / "CHECKS.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:1]))
print(f"wrote results.npz, numbers.json, CHECKS.txt in {OUT['runtime_s']['total']:.0f}s")
