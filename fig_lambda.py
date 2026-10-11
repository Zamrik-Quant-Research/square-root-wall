"""fig_lambda — the tail exponents lambda(c) (two-sided wall) and lambda+(c) (one-sided wall) across c."""
from pathlib import Path

import numpy as np

import wall_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
c, lam, lamp = R["c_grid"], R["lam_grid"], R["lam_plus_grid"]
cp = dict(zip(R["p_list"], R["c_p"]))
YMAX = 5.0

fig, ax = P.figure()
ax.plot(c, np.clip(lam, None, YMAX + 1), color=P.NAVY, lw=2.6, label=r"two-sided wall, $\lambda(c)$")
ax.plot(c, lamp, color=P.TEAL, lw=2.6, label=r"one-sided wall, $\lambda^{+}(c)$")
small = c[c > 0.42]
ax.plot(small, np.pi ** 2 / (8 * small ** 2), color=P.GREY, lw=1.3, ls=":",
        label=r"short-wall limit $\pi^2/(8c^2)$")
for p, name in ((0.5, r"$\frac{1}{2}$"), (1.0, "1"), (2.0, "2")):
    ax.axhline(p, color=P.GREY, lw=1.0, ls="--")
    ax.plot(cp[p], p, "o", ms=8, color=P.RED, mec="white", mew=1.4, zorder=5)
    ax.annotate(rf"$c_{{{name if p != 0.5 else '1/2'}}} = {cp[p]:.3f}$", xy=((cp[p] + 0.05, p + 0.09) if p > 0.5 else (cp[p] - 0.45, p - 0.20)),
                color=P.NAVY, fontsize=12)
ax.annotate(r"$\mathbb{E}[(1+\tau_c)^p] < \infty$ exactly when $p < \lambda(c)$", xy=(1.55, 3.3),
            color=P.GREY, fontsize=12)
ax.annotate(r"$\lambda^{+}(c) < \frac{1}{2}$ for every $c > 0$", xy=(1.85, 0.62), color=P.TEAL, fontsize=12)
ax.set_xlim(0, 3.0); ax.set_ylim(0, YMAX)
P.label(ax, x=r"wall height $c$ (standard deviations of the path's age)", y=r"tail exponent")
P.legend(ax, loc="upper right")
P.save(fig, "fig_lambda.png")
