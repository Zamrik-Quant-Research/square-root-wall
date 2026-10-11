"""fig_clock — six Brownian paths against the wall c sqrt(1+t), c = 1, in clock time and in log time."""
from pathlib import Path

import numpy as np

import wall_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
t, B, first = R["clock_t"], R["clock_B"], R["clock_first"]
s = np.log1p(t)
U = B / np.sqrt(1.0 + t)
C = 1.0

fig, (ax1, ax2) = P.panels(2)
wall_t = C * np.sqrt(1.0 + t)
for ax, x, w in ((ax1, t, wall_t), (ax2, s, np.full_like(s, C))):
    ax.plot(x, w, color=P.NAVY, lw=2.6)
    ax.plot(x, -w, color=P.NAVY, lw=2.6)
for i in range(B.shape[0]):
    end = first[i] + 1 if first[i] >= 0 else t.size
    alpha = 0.85 if first[i] >= 0 else 0.55
    ax1.plot(t[:end], B[i, :end], color=P.TEAL, lw=0.9, alpha=alpha)
    ax2.plot(s[:end], U[i, :end], color=P.TEAL, lw=0.9, alpha=alpha)
    if first[i] >= 0:
        k, sg = first[i], np.sign(B[i, first[i]])          # the stop, marked on the wall it reached
        ax1.plot(t[k], sg * wall_t[k], "o", ms=7, color=P.RED, mec="white", mew=1.2, zorder=5)
        ax2.plot(s[k], sg * C, "o", ms=7, color=P.RED, mec="white", mew=1.2, zorder=5)
ax1.annotate(r"$\pm\sqrt{1+t}$", xy=(22.5, 5.4), color=P.NAVY, fontsize=12.5)
ax2.annotate(r"$\pm 1$", xy=(3.05, 1.12), color=P.NAVY, fontsize=12.5)
ax1.set_xlim(0, t[-1]); ax1.set_ylim(-6.6, 6.6)
ax2.set_xlim(0, s[-1]); ax2.set_ylim(-1.65, 1.65)
ax1.set_title(r"clock time: $B_t$ against the moving wall", color=P.GREY, fontsize=12.5)
ax2.set_title(r"log time: $U_s = B_t/\sqrt{1+t}$ against a fixed wall", color=P.GREY, fontsize=12.5)
P.label(ax1, x=r"time $t$", y="position")
P.label(ax2, x=r"log time $s = \log(1+t)$", y=r"position in units of $\sqrt{1+t}$")
P.save(fig, "fig_clock.png")
