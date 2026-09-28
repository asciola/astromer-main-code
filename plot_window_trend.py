#!/usr/bin/env python3
"""How does out-of-fold R^2 depend on window size and latent size?

Reads one summary.csv from fit_regressor.py and draws, for each target (z, m, e):

  left   a latent x window grid. Each cell shows the model's R^2; the color is its
         gain over the summary-statistics control in the same run (teal = better
         than the control, brown = worse, gray = about the same), so the grid
         answers "which models add anything" at a glance.
  right  R^2 against window size, one line per latent. All latents are drawn in
         gray; the ones named by --highlight (default 4 and 16) are colored and
         labeled, and the control is the dashed line.

Model names are parsed for latent and window:
    ..._K_01_<latent>_<window>          -> latent <latent>
    ..._base_256_<window>_K_baseline    -> latent "base" (no latent bottleneck)
Anything else is skipped with a note.

    python plot_window_trend.py --summary agn_mlp_stats/summary.csv \\
        --title "MLP, statistics + embeddings" --out window_trend_mlp_stats.png
    python plot_window_trend.py --summary agn_mlp_stats/summary.csv --highlight 4 16 128
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED, RULE = "#101A19", "#384645", "#5F6B6A", "#DBE1E0"
ACCENT, ACCENT2 = "#0E8C7E", "#A06A00"
LINE_GRAY = "#B9C1C0"
# diverging: brown (worse than control) - neutral gray - teal (better)
DIVERGE = LinearSegmentedColormap.from_list(
    "gain", ["#7A4E0B", "#C9A36A", "#EEEFEE", "#7CC3B9", "#06574E"])
TARGETS = [("z", "redshift  z"), ("m", r"$\log M_{\rm BH}$"), ("e", r"$\log L/L_{\rm Edd}$")]
CONTROL = "CONTROL: summary stats only"

PAT_LATENT = re.compile(r"_K_01_(\d+)_(\d+)$")
PAT_BASE = re.compile(r"_base_\d+_(\d+)_K_baseline$")


def parse(name: str):
    m = PAT_LATENT.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = PAT_BASE.search(name)
    if m:
        return "base", int(m.group(1))
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True, help="summary.csv from fit_regressor.py")
    ap.add_argument("--targets", default="z,m,e")
    ap.add_argument("--highlight", nargs="+", default=["4", "16"],
                    help="latents to color in the line panels (default 4 16; 'base' allowed)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    ctrl = s[s["model"] == CONTROL]
    ctrl = ctrl.iloc[0] if len(ctrl) else None
    rows, skipped = [], []
    for _, r in s.iterrows():
        if r["model"] == CONTROL:
            continue
        lat, ws = parse(str(r["model"]))
        if lat is None:
            skipped.append(r["model"])
            continue
        rows.append({**r.to_dict(), "latent": lat, "window": ws})
    if skipped:
        print(f"note: skipped {len(skipped)} model(s) with unrecognised names: {skipped[:3]}...")
    d = pd.DataFrame(rows)
    if d.empty:
        print("no models parsed", file=sys.stderr)
        return 1

    lat_num = sorted(v for v in d["latent"].unique() if v != "base")
    latents = lat_num + (["base"] if (d["latent"] == "base").any() else [])
    windows = sorted(d["window"].unique())
    tgts = [(k, lab) for k, lab in TARGETS if k in args.targets.split(",") and f"r2_{k}" in d]
    hl = [int(h) if h.isdigit() else h for h in args.highlight]
    hl_colors = {h: c for h, c in zip(hl, [ACCENT, ACCENT2])}
    if len(hl) > 2:
        print("note: only the first two --highlight latents get a color; the rest stay gray")

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    n = len(tgts)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 3.9 * n),
                             gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42,
                                          "hspace": 0.55}, squeeze=False)
    print(f"{args.summary}: {len(d)} models, latents {latents}, windows {windows}")

    for row, (k, lab) in enumerate(tgts):
        col = f"r2_{k}"
        c0 = float(ctrl[col]) if ctrl is not None else np.nan
        grid = np.full((len(latents), len(windows)), np.nan)
        for _, r in d.iterrows():
            grid[latents.index(r["latent"]), windows.index(r["window"])] = r[col]
        gain = grid - c0 if np.isfinite(c0) else grid - np.nanmean(grid)

        # ---- heatmap -----------------------------------------------------------
        ax = axes[row, 0]
        lim = max(np.nanmax(np.abs(gain)), 1e-3)
        im = ax.imshow(gain, cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(len(latents)):
            for j in range(len(windows)):
                v = grid[i, j]
                if not np.isfinite(v):
                    continue
                dark = abs(gain[i, j]) > 0.6 * lim
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.4,
                        color="white" if dark else INK)
        best = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                                   ec=INK, lw=1.6))
        ax.set_xticks(range(len(windows)), [str(w) for w in windows])
        ax.set_yticks(range(len(latents)), [str(l) for l in latents])
        ax.set_xlabel("window size")
        ax.set_ylabel("latent size")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ctrl_txt = f"control R$^2$ = {c0:.3f}" if np.isfinite(c0) else "no control row"
        ax.set_title(f"{lab}:  R$^2$ per model  (color = gain over control; {ctrl_txt})",
                     loc="left", fontsize=9, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label("R$^2$ $-$ control", fontsize=8, color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED)

        # ---- lines -------------------------------------------------------------
        ax = axes[row, 1]
        order = [l for l in latents if l not in hl_colors] + [l for l in latents if l in hl_colors]
        end_labels = []
        for lat in order:
            sub = d[d["latent"] == lat].sort_values("window")
            if sub.empty:
                continue
            x, y = sub["window"].to_numpy(), sub[col].to_numpy()
            if lat in hl_colors:
                c = hl_colors[lat]
                ax.plot(x, y, "-o", color=c, lw=2.0, ms=5.5, mec="white", mew=1.2, zorder=4)
                end_labels.append([y[-1], x[-1], f"latent {lat}"])
            else:
                ls = (0, (3, 2)) if lat == "base" else "-"
                ax.plot(x, y, ls=ls, color=LINE_GRAY, lw=1.1, zorder=2)
        # end-of-line labels, nudged apart so they never overlap
        if end_labels:
            ylo, yhi = ax.get_ylim()
            gap = 0.06 * (yhi - ylo)
            end_labels.sort()
            for i in range(1, len(end_labels)):
                if end_labels[i][0] - end_labels[i - 1][0] < gap:
                    end_labels[i][0] = end_labels[i - 1][0] + gap
            for ylab, xlab, txt in end_labels:
                ax.annotate(txt, (xlab, ylab), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
        if np.isfinite(c0):
            ax.axhline(c0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
            ax.annotate("control", (windows[0], c0), xytext=(0, 4), textcoords="offset points",
                        fontsize=7.8, color=MUTED)
        ax.set_xticks(windows)
        ax.set_xlim(windows[0] - 50, windows[-1] + 170)
        ax.set_xlabel("window size")
        ax.set_ylabel("out-of-fold R$^2$")
        ax.set_title(f"{lab}:  R$^2$ vs window  (gray = other latents"
                     + (", dashed = base" if "base" in latents else "") + ")",
                     loc="left", fontsize=9, color=INK)
        ax.grid(True, color=RULE, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        bi, bj = best
        print(f"  {k}: best latent {latents[bi]}, window {windows[bj]}, R2 {grid[bi, bj]:.3f}"
              + (f"  (control {c0:.3f}, gain {grid[bi, bj] - c0:+.3f})" if np.isfinite(c0) else ""))
        for lat in hl:
            if lat in latents:
                v = grid[latents.index(lat)]
                trend = ", ".join(f"{w}:{x:.3f}" for w, x in zip(windows, v) if np.isfinite(x))
                print(f"     latent {lat}: {trend}")

    head = str(d["head"].iloc[0]) if "head" in d else ""
    ttl = args.title or f"{args.summary.parent.name}  ({head})"
    fig.subplots_adjust(top=0.93)
    fig.suptitle(ttl, y=0.975, fontsize=11.5, color=INK)
    fig.text(0.5, -0.01, "Boxed cell = best model for that target. Gains are relative to the "
             "summary-statistics control fitted with the same head in the same run.",
             ha="center", va="top", fontsize=7.8, color=MUTED)
    out = args.out or Path(f"window_trend_{args.summary.parent.name}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
