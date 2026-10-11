"""fig_bracket — the win rate of a bracket that widens with sqrt(1+t), against the fixed bracket's a/(a+b)."""
from pathlib import Path

import numpy as np

import wall_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
b = R["b_grid"]

fig, ax = P.figure()
for a, key, col, ls in ((0.5, "win_a05", P.TEAL, "-"), (1.0, "win_a1", P.NAVY, "-"), (2.0, "win_a2", P.GREY, "-")):
    ax.plot(b, R[key], color=col, lw=2.6, ls=ls, label=rf"widening bracket, stop $a = {a:g}$")
ax.plot(b, 1.0 / (1.0 + b), color=P.NAVY, lw=1.6, ls="--", label=r"fixed bracket, $a/(a+b)$ with $a = 1$")
ax.errorbar(R["mc_b"], R["mc_win"], yerr=3 * R["mc_win_se"], fmt="s", ms=7, color=P.NAVY, mfc="white",
            mec=P.NAVY, mew=1.6, ecolor=P.NAVY, capsize=3, lw=1.2, zorder=5,
            label="simulation, $10^6$ paths each ($\\pm 3$ s.e.)")
for a_, b_, w_ in zip(R["mc_a"], R["mc_b"], R["mc_win"]):
    dx, dy = {(2.0, 1.0): (0.06, -0.075), (1.0, 3.0): (-0.4, 0.07)}.get((a_, b_), (0.06, 0.045))
    ax.annotate(f"({a_:g}, {b_:g})", xy=(b_ + dx, w_ + dy), color=P.GREY, fontsize=11)
ax.set_xlim(0.25, 4.0); ax.set_ylim(0, 1.0)
P.label(ax, x=r"target $b$ (in units of $\sqrt{1+t}$)", y="probability the target is hit first")
P.legend(ax, loc="upper right")
P.save(fig, "fig_bracket.png")
