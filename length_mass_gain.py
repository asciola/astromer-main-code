#!/usr/bin/env python3
"""Does a longer window and/or a latent bottleneck (MHLA) help more on longer light curves,
and more for heavier black holes?

Reads the out-of-fold predictions.csv from an EMBEDDINGS-ONLY fit_regressor.py run (no
--with-stats; agn_ridge or agn_mlp) and lc_index.csv, bins the sources by light-curve
length (and by true mass), and compares every model with the baseline model (default:
the no-bottleneck model at window 200) inside each bin. Writes:

  length_breakdown.csv         how many sources fall in each length bin (Pavlos' <200,
                               200-500, 500+ and the finer bins used for the plots)
  gain_by_length.csv           per model x length bin (x mass bin): n, R^2, MSE, ratios, CIs
  gain_vs_length_<t>.png       (3) R^2(model) / R^2(base) against length. One panel per
                               latent family (base, then --latents); one line per window.
  gain_vs_length_mass_<t>.png  (4) the same comparison split by true-mass tercile, one
                               panel per --focus model, one line per tercile.
  gain_length_mass_map_<t>.png (5) length x mass grid for each --focus model.
  gain_vs_length_mass_excess_<t>.png
                               (4b) figure 4 divided by what a UNIFORM improvement would
                               give in each tercile (see NULL below): 1 = no mass effect.
  prediction_shift_<t>.png     how far each --focus model's mean prediction moves from the
                               base's, and each model's mean residual, per length bin.

NULL (why figure 4 alone misleads)
  Splitting by TRUE mass rewards any model that shrinks less toward the mean: the base
  predicts nearly the average mass, which is almost right for the middle tercile, so a
  model that is better overall looks worse there and better at both ends even if its
  gain has nothing to do with mass. For each length bin the script simulates that case
  using the bin's real masses and the two models' real R^2 values, and figure 4b shows
  observed / expected. A mean shift of the predictions with length (figure 6) is the
  other thing that can make the high and low terciles move in opposite directions.

WHY TWO METRICS
  R^2 ratio   what was asked for. Within a LENGTH bin the target keeps its full spread,
              so R^2 is meaningful there.
  MSE ratio   MSE(base) / MSE(model), > 1 means the model is better. Used for anything
              split by MASS: inside a mass tercile the mass range is cut to a third, R^2
              there is close to zero or negative for every model, and a ratio of two such
              numbers is unstable or meaningless. The MSE ratio has no such problem.
  Both ratios read the same way (> 1 = the larger model helps), and on length-only bins
  the MSE ratio is also drawn as a check. The bands are paired bootstrap intervals over
  sources (default 68%, i.e. +/-1 sigma; --ci 95 for 95%).

LENGTH
  --length r (default) number of r-band points; also g, i, max (longest band), sum.
  Only ~5% of sources have fewer than 200 r-band points, so Pavlos' three buckets are
  far from equal; the plots use finer bins (--length-bins) to show the shape.

    python length_mass_gain.py --pred agn_ridge/predictions.csv --index $IDX --out-dir lengain_ridge
    python length_mass_gain.py --pred agn_mlp/predictions.csv   --index $IDX --out-dir lengain_mlp
    python length_mass_gain.py ... --target z --latents 4 16 --focus K_01_16_800 base_256_800
"""
from __future__ import annotations

import argparse
import re
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED, RULE = "#101A19", "#384645", "#5F6B6A", "#DBE1E0"
WINDOW_RAMP = ["#48C1B0", "#129E8E", "#007D6E", "#005C4F", "#003D32"]    # ordinal, validated
MASS_RAMP = ["#E99E5E", "#B06A26", "#7A3800"]                             # low -> high mass
DIVERGE = LinearSegmentedColormap.from_list(
    "gain", ["#7A4E0B", "#C9A36A", "#EEEFEE", "#7CC3B9", "#06574E"])
TLABEL = {"m": r"$\log M_{\rm BH}$", "z": "redshift z", "e": r"$\log L/L_{\rm Edd}$"}
PAT_LATENT = re.compile(r"_K_01_(\d+)_(\d+)$")
PAT_BASE = re.compile(r"_base_\d+_(\d+)_K_baseline$")
CONTROL = "CONTROL: summary stats only"
PAVLOS_BINS = [0, 200, 500, np.inf]


def parse(name):
    m = PAT_LATENT.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m = PAT_BASE.search(name)
    if m:
        return "base", int(m.group(1))
    return None, None


def bin_labels(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f"<{int(b)}" if a == 0 else (f"{int(a)}+" if not np.isfinite(b) else f"{int(a)}-{int(b)}"))
    return out


def weights(n, B, rng):
    """(B, n) bootstrap multiplicities."""
    W = np.empty((B, n), dtype=np.float32)
    for i in range(B):
        W[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def stats_with_ci(y, p_m, p_b, B, rng, lo_q, hi_q):
    """Point estimates and bootstrap intervals for R^2 ratio and MSE ratio (paired)."""
    n = len(y)
    em, eb = (p_m - y) ** 2, (p_b - y) ** 2
    var = y.var()
    r2m = 1 - em.mean() / var if var > 0 else np.nan
    r2b = 1 - eb.mean() / var if var > 0 else np.nan
    out = {"n": n, "r2": r2m, "r2_base": r2b, "mse": em.mean(), "mse_base": eb.mean(),
           "r2_ratio": r2m / r2b if r2b > 0 else np.nan,
           "mse_ratio": eb.mean() / em.mean(),
           # mean residuals, and how far the model's predictions sit from the base's
           "bias": float((p_m - y).mean()), "bias_base": float((p_b - y).mean()),
           "shift": float((p_m - p_b).mean()),
           "shift_se": float((p_m - p_b).std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan}
    if B > 0 and n >= 20:
        W = weights(n, B, rng)
        sw = W.sum(1)
        sy, syy = W @ y, W @ (y * y)
        v = syy / sw - (sy / sw) ** 2
        mm, mb = (W @ em) / sw, (W @ eb) / sw
        r2m_b, r2b_b = 1 - mm / v, 1 - mb / v
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(r2b_b > 0, r2m_b / r2b_b, np.nan)
        mr = mb / mm
        out.update(r2_ratio_lo=np.nanpercentile(rr, lo_q), r2_ratio_hi=np.nanpercentile(rr, hi_q),
                   mse_ratio_lo=np.percentile(mr, lo_q), mse_ratio_hi=np.percentile(mr, hi_q),
                   r2_ratio_unstable=float(np.mean(~(r2b_b > 0))))
    return out


def null_tercile_ratios(y, labels, n_groups, r2_base, r2_model, rng, draws=20):
    """Expected MSE(base)/MSE(model) per mass group if the model were UNIFORMLY better.

    Both predictors are least-squares fits to a synthetic feature that carries a fixed
    fraction of the variance of the real y values in this bin (R^2 = r2_base for the
    base, r2_model for the model) and nothing else: no dependence on mass or length.
    Splitting such predictions by true mass still gives the middle group a ratio below 1
    and the outer groups above 1; that is the pattern to subtract.
    """
    if not (r2_base > 0 and r2_model > 0) or len(y) < 50:
        return np.full(n_groups, np.nan)
    z = (y - y.mean()) / y.std()
    acc = np.zeros((draws, n_groups))
    for d in range(draws):
        preds = []
        for r2 in (r2_base, r2_model):
            f = np.sqrt(r2) * z + np.sqrt(1 - r2) * rng.normal(size=len(y))
            b = np.polyfit(f, y, 1)
            preds.append(np.polyval(b, f))
        eb, em = (preds[0] - y) ** 2, (preds[1] - y) ** 2
        for k in range(n_groups):
            sk = labels == k
            acc[d, k] = eb[sk].mean() / em[sk].mean() if sk.sum() >= 20 else np.nan
    return np.nanmean(acc, axis=0)


def style(ax):
    ax.grid(True, color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True,
                    help="predictions.csv from an embeddings-only run (agn_ridge or agn_mlp)")
    ap.add_argument("--index", type=Path, required=True, help="lc_index.csv")
    ap.add_argument("--target", default="m", choices=["m", "z", "e"])
    ap.add_argument("--base", default="base_256_200_K_baseline",
                    help="substring naming the reference model (default: base, window 200)")
    ap.add_argument("--length", default="r", choices=["r", "g", "i", "max", "sum"])
    ap.add_argument("--length-bins", type=float, nargs="+",
                    default=[0, 200, 300, 400, 500, 600, 700, 800, np.inf])
    ap.add_argument("--latents", nargs="+", default=["4", "16", "128"],
                    help="latent families to draw next to base in figure 3")
    ap.add_argument("--focus", nargs="+",
                    default=["base_256_800_K", "K_01_16_200", "K_01_4_800", "K_01_16_800"],
                    help="models (substrings) for figures 4 and 5")
    ap.add_argument("--mass-bins", type=int, default=3, help="quantile bins of true mass (default 3)")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--null-draws", type=int, default=20,
                    help="simulations per bin for the uniform-improvement null (default 20)")
    ap.add_argument("--ci", type=float, default=68.0, help="interval width in percent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("length_gain"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lo_q, hi_q = 50 - a.ci / 2, 50 + a.ci / 2

    # ---- lengths and masses per source ------------------------------------------
    idx = pd.read_csv(a.index, dtype={"source_id": str})
    npts = idx.pivot_table(index="source_id", columns="band_name", values="n_points",
                           aggfunc="first").fillna(0)
    length = {"max": npts.max(axis=1), "sum": npts.sum(axis=1)}.get(
        a.length, npts[a.length] if a.length in npts else None)
    if length is None:
        print(f"no band {a.length!r} in lc_index.csv", file=sys.stderr)
        return 1
    per_src = idx.drop_duplicates("source_id").set_index("source_id")

    # ---- predictions ------------------------------------------------------------
    tcol, pcol = f"{a.target}_true", f"{a.target}_pred"
    pred = pd.read_csv(a.pred, dtype={"source_id": str}, usecols=lambda c: c in
                       {"source_id", "model", "window_size", tcol, pcol, "m_true"})
    pred = pred[pred["model"] != CONTROL]
    names = sorted(pred["model"].unique())
    base_hits = [n for n in names if a.base in n]
    if len(base_hits) != 1:
        print(f"--base {a.base!r} matched {base_hits or 'nothing'}; models are:\n  " +
              "\n  ".join(names), file=sys.stderr)
        return 1
    base = base_hits[0]
    wide = pred.pivot_table(index="source_id", columns="model", values=pcol, aggfunc="mean")
    y = pred.drop_duplicates("source_id").set_index("source_id")[tcol].reindex(wide.index)
    mass_true = per_src["LOGMBH"].reindex(wide.index)
    L = length.reindex(wide.index)
    ok = y.notna() & L.notna() & wide[base].notna()
    wide, y, L, mass_true = wide[ok], y[ok].to_numpy(float), L[ok].to_numpy(float), mass_true[ok]
    print(f"{len(y):,} sources with predictions; reference model: {base}")

    # ---- step 1: how the lengths break down -------------------------------------
    rows = []
    for scheme, edges in (("Pavlos", PAVLOS_BINS), ("plot bins", a.length_bins)):
        labs = bin_labels(edges)
        c = pd.cut(L, edges, right=False, labels=labs).value_counts().reindex(labs)
        for lab, v in c.items():
            rows.append({"scheme": scheme, "length_measure": a.length, "bin": lab,
                         "n_sources": int(v), "fraction": v / len(L)})
    lb = pd.DataFrame(rows)
    lb.to_csv(a.out_dir / "length_breakdown.csv", index=False)
    print(f"\nlength ({a.length}) breakdown:")
    for scheme, g in lb.groupby("scheme", sort=False):
        print(f"  {scheme:10s} " + "  ".join(f"{r.bin}: {r.n_sources:,} ({r.fraction:.1%})"
                                             for r in g.itertuples()))

    # ---- per-bin comparison ------------------------------------------------------
    edges = a.length_bins
    labs = bin_labels(edges)
    lbin = pd.cut(L, edges, right=False, labels=labs).astype(str)
    mq = pd.qcut(mass_true, a.mass_bins, labels=False, duplicates="drop").to_numpy()
    medges = np.quantile(mass_true.dropna(), np.linspace(0, 1, a.mass_bins + 1))
    mlabs = [f"{medges[k]:.2f}-{medges[k+1]:.2f}" for k in range(a.mass_bins)]
    mnames = (["low", "mid", "high"] if a.mass_bins == 3 else [f"q{k+1}" for k in range(a.mass_bins)])

    pb = wide[base].to_numpy(float)
    out = []
    models = [n for n in names if parse(n)[0] is not None]
    focus = []
    for f in a.focus:
        hits = [n for n in models if f in n]
        if len(hits) == 1:
            focus.append(hits[0])
        else:
            print(f"note: --focus {f!r} matched {len(hits)} models; skipped")
    for mdl in models:
        lat, ws = parse(mdl)
        pm = wide[mdl].to_numpy(float)
        good = np.isfinite(pm)
        do_mass = mdl in focus
        for lab in labs:
            s = good & (lbin == lab)
            if s.sum() < 20:
                continue
            r = stats_with_ci(y[s], pm[s], pb[s], a.bootstrap, rng, lo_q, hi_q)
            out.append({"model": mdl, "latent": lat, "window": ws, "length_bin": lab,
                        "mass_bin": "all", **r})
            if do_mass:
                null = null_tercile_ratios(y[s], mq[s], a.mass_bins, r["r2_base"], r["r2"],
                                           rng, a.null_draws)
                for k in range(a.mass_bins):
                    sk = s & (mq == k)
                    if sk.sum() < 20:
                        continue
                    r = stats_with_ci(y[sk], pm[sk], pb[sk], a.bootstrap, rng, lo_q, hi_q)
                    nk = null[k]
                    r.update(null_mse_ratio=nk,
                             excess=r["mse_ratio"] / nk if np.isfinite(nk) else np.nan)
                    if "mse_ratio_lo" in r and np.isfinite(nk):
                        r.update(excess_lo=r["mse_ratio_lo"] / nk, excess_hi=r["mse_ratio_hi"] / nk)
                    out.append({"model": mdl, "latent": lat, "window": ws, "length_bin": lab,
                                "mass_bin": mnames[k], "mass_range": mlabs[k], **r})
    res = pd.DataFrame(out)
    res.to_csv(a.out_dir / "gain_by_length.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    x = np.arange(len(labs))
    counts = pd.Series(lbin).value_counts().reindex(labs).fillna(0).astype(int)
    tl = TLABEL[a.target]
    base_short = f"base, window {parse(base)[1]}"

    # ---- figure 3: ratio vs length, per latent family ----------------------------
    fams = ["base"] + [l for l in a.latents if l != "base"]
    fams = [f for f in fams if (res["latent"] == f).any()]
    for metric, ylab, fname in (("r2_ratio", f"R$^2$(model) / R$^2$({base_short})", "gain_vs_length"),
                                ("mse_ratio", f"MSE({base_short}) / MSE(model)", "gain_vs_length_mse")):
        fig, axes = plt.subplots(1, len(fams), figsize=(4.4 * len(fams), 4.2), sharey=True,
                                 squeeze=False)
        allv = res[(res["mass_bin"] == "all")][metric]
        for ax, fam in zip(axes[0], fams):
            sub = res[(res["latent"] == fam) & (res["mass_bin"] == "all")]
            wins = sorted(sub["window"].unique())
            for w in wins:
                d = sub[sub["window"] == w].set_index("length_bin").reindex(labs)
                if parse(base) == (fam, w):
                    continue
                c = WINDOW_RAMP[min(len(WINDOW_RAMP) - 1, [200, 400, 600, 800, 1000].index(w)
                                    if w in (200, 400, 600, 800, 1000) else -1)]
                if f"{metric}_lo" in d:
                    ax.fill_between(x, d[f"{metric}_lo"], d[f"{metric}_hi"], color=c, alpha=0.12, lw=0)
                ax.plot(x, d[metric], "-o", color=c, lw=1.8, ms=4.5, mec="white", mew=1, label=f"window {w}")
            ax.axhline(1, color=MUTED, lw=1.1, ls=(0, (4, 3)))
            ax.set_xticks(x, labs, rotation=35, ha="right")
            ax.set_title("no latent bottleneck (base)" if fam == "base" else f"MHLA, latent {fam}",
                         loc="left", fontsize=9.5)
            ax.set_xlabel(f"{a.length}-band points per light curve")
            style(ax)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        axes[0][0].set_ylabel(ylab)
        for j, n in enumerate(counts):
            axes[0][0].annotate(f"{n:,}", (j, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
        fig.suptitle(f"{tl}: gain over {base_short}, by light-curve length  "
                     f"(> 1 = better; band = {a.ci:.0f}% bootstrap interval; numbers = sources per bin)",
                     y=1.02, fontsize=10.5)
        fig.tight_layout()
        p = a.out_dir / f"{fname}_{a.target}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")

    # ---- figure 4: ratio vs length, split by mass --------------------------------
    if focus:
        fig, axes = plt.subplots(1, len(focus), figsize=(4.4 * len(focus), 4.2), sharey=True,
                                 squeeze=False)
        for ax, mdl in zip(axes[0], focus):
            lat, ws = parse(mdl)
            for k, mn in enumerate(mnames):
                d = res[(res["model"] == mdl) & (res["mass_bin"] == mn)].set_index("length_bin").reindex(labs)
                c = MASS_RAMP[k] if a.mass_bins == 3 else None
                ax.fill_between(x, d["mse_ratio_lo"], d["mse_ratio_hi"], color=c, alpha=0.13, lw=0)
                ax.plot(x, d["mse_ratio"], "-o", color=c, lw=1.8, ms=4.5, mec="white", mew=1,
                        label=f"{mn} mass ({mlabs[k]})")
            ax.axhline(1, color=MUTED, lw=1.1, ls=(0, (4, 3)))
            ax.set_xticks(x, labs, rotation=35, ha="right")
            ax.set_title(("base" if lat == "base" else f"MHLA latent {lat}") + f", window {ws}",
                         loc="left", fontsize=9.5)
            ax.set_xlabel(f"{a.length}-band points per light curve")
            style(ax)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        axes[0][0].set_ylabel(f"MSE({base_short}) / MSE(model)")
        fig.suptitle(f"{tl}: gain over {base_short} by length, split by true-mass tercile  "
                     f"(MSE ratio; > 1 = better)", y=1.02, fontsize=10.5)
        fig.tight_layout()
        p = a.out_dir / f"gain_vs_length_mass_{a.target}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")

        # ---- figure 5: length x mass map -------------------------------------------
        fig, axes = plt.subplots(1, len(focus), figsize=(4.6 * len(focus), 3.6), squeeze=False)
        grids = []
        for mdl in focus:
            g = np.full((a.mass_bins, len(labs)), np.nan)
            n = np.zeros_like(g)
            for k, mn in enumerate(mnames):
                d = res[(res["model"] == mdl) & (res["mass_bin"] == mn)].set_index("length_bin").reindex(labs)
                g[k] = d["mse_ratio"].to_numpy(float)
                n[k] = d["n"].fillna(0).to_numpy(float)
            grids.append((g, n))
        lim = max(np.nanmax(np.abs(g - 1)) for g, _ in grids)
        for j_ax, (ax, mdl, (g, n)) in enumerate(zip(axes[0], focus, grids)):
            lat, ws = parse(mdl)
            im = ax.imshow(g, origin="lower", aspect="auto", cmap=DIVERGE,
                           norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim))
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    if np.isfinite(g[i, j]):
                        dark = abs(g[i, j] - 1) > 0.6 * lim
                        ax.text(j, i, f"{g[i, j]:.3f}\n{int(n[i, j]):,}", ha="center", va="center",
                                fontsize=6.3, color="white" if dark else INK, linespacing=1.2)
            ax.set_xticks(range(len(labs)), labs, rotation=35, ha="right")
            ax.set_yticks(range(a.mass_bins),
                          [f"{m}\n{r}" for m, r in zip(mnames, mlabs)] if j_ax == 0 else [])
            ax.set_xlabel(f"{a.length}-band points per light curve")
            ax.set_title(("base" if lat == "base" else f"MHLA latent {lat}") + f", window {ws}",
                         loc="left", fontsize=9.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0][0].set_ylabel(r"true $\log M_{\rm BH}$ tercile")
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(f"MSE({base_short}) / MSE(model)", fontsize=8)
        cb.outline.set_visible(False)
        fig.suptitle(f"{tl}: gain over {base_short} by length and mass  "
                     "(cell: MSE ratio, sources)", y=1.03, fontsize=10.5)
        p = a.out_dir / f"gain_length_mass_map_{a.target}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")

    if focus:
        # ---- figure 4b: gain beyond the uniform-improvement null ---------------------
        fig, axes = plt.subplots(1, len(focus), figsize=(4.4 * len(focus), 4.2), sharey=True,
                                 squeeze=False)
        for ax, mdl in zip(axes[0], focus):
            lat, ws = parse(mdl)
            for k, mn in enumerate(mnames):
                d = res[(res["model"] == mdl) & (res["mass_bin"] == mn)].set_index("length_bin").reindex(labs)
                c = MASS_RAMP[k] if a.mass_bins == 3 else None
                if "excess_lo" in d:
                    ax.fill_between(x, d["excess_lo"], d["excess_hi"], color=c, alpha=0.13, lw=0)
                ax.plot(x, d["excess"], "-o", color=c, lw=1.8, ms=4.5, mec="white", mew=1,
                        label=f"{mn} mass ({mlabs[k]})")
            ax.axhline(1, color=MUTED, lw=1.1, ls=(0, (4, 3)))
            ax.set_xticks(x, labs, rotation=35, ha="right")
            ax.set_title(("base" if lat == "base" else f"MHLA latent {lat}") + f", window {ws}",
                         loc="left", fontsize=9.5)
            ax.set_xlabel(f"{a.length}-band points per light curve")
            style(ax)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        axes[0][0].set_ylabel("observed / expected MSE ratio")
        fig.suptitle(f"{tl}: gain over {base_short} BEYOND a uniform improvement, by mass tercile  "
                     "(1 = no mass-specific effect; > 1 = this tercile gains more than expected)",
                     y=1.02, fontsize=10.5)
        fig.tight_layout()
        p = a.out_dir / f"gain_vs_length_mass_excess_{a.target}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")

        # ---- figure 6: prediction shift and mean residual vs length ------------------
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        shift_colors = [WINDOW_RAMP[1], "#A06A00", WINDOW_RAMP[3], "#5F6B6A"]
        allrows = res[res["mass_bin"] == "all"]
        for j, mdl in enumerate(focus):
            lat, ws = parse(mdl)
            d = allrows[allrows["model"] == mdl].set_index("length_bin").reindex(labs)
            c = shift_colors[j % len(shift_colors)]
            name = ("base" if lat == "base" else f"latent {lat}") + f", window {ws}"
            a1.errorbar(x + (j - 1.5) * 0.06, d["shift"], yerr=d["shift_se"], fmt="-o", ms=4.5,
                        lw=1.6, color=c, mec="white", mew=1, capsize=0, label=name)
            a2.plot(x, d["bias"], "-o", ms=4.5, lw=1.6, color=c, mec="white", mew=1, label=name)
        d0 = allrows[allrows["model"] == focus[0]].set_index("length_bin").reindex(labs)
        a2.plot(x, d0["bias_base"], "--s", ms=4, lw=1.4, color=INK2, label=base_short)
        for ax, yl, ttl in ((a1, f"mean(model $-$ {base_short}) prediction",
                             "a   do the models shift their predictions with length?"),
                            (a2, "mean residual (predicted $-$ true)",
                             "b   mean residual by length (0 = unbiased in that bin)")):
            ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
            ax.set_xticks(x, labs, rotation=35, ha="right")
            ax.set_xlabel(f"{a.length}-band points per light curve")
            ax.set_ylabel(yl)
            ax.set_title(ttl, loc="left", fontsize=9.5)
            style(ax)
        a1.legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"{tl}: a shift up for long curves would help the high tercile and hurt the low one "
                     "without adding information", y=1.03, fontsize=10)
        fig.tight_layout()
        p = a.out_dir / f"prediction_shift_{a.target}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")

        print("\nobserved / expected (uniform-improvement null) MSE ratio, by tercile:")
        for mdl in focus:
            print(f"  {mdl[-26:]}")
            t = res[(res["model"] == mdl) & (res["mass_bin"] != "all")].pivot_table(
                index="mass_bin", columns="length_bin", values="excess").reindex(mnames)
            print("    " + t.reindex(columns=labs).round(3).to_string().replace("\n", "\n    "))
        print("\nprediction shift, mean(model - base), by length bin:")
        for mdl in focus:
            d = allrows[allrows["model"] == mdl].set_index("length_bin").reindex(labs)
            print(f"  {mdl[-26:]:>26}: " + "  ".join(f"{v:+.3f}" for v in d["shift"]))

    # ---- console summary: is there an upward trend? -------------------------------
    print("\nslope of the MSE ratio across length bins (per 100 points, weighted by bin size):")
    centers = np.array([(e0 + (e1 if np.isfinite(e1) else e0 + 200)) / 2
                        for e0, e1 in zip(edges[:-1], edges[1:])])
    for mdl in [base] + focus:
        if mdl == base:
            continue
        for mn in ["all"] + (mnames if mdl in focus else []):
            d = res[(res["model"] == mdl) & (res["mass_bin"] == mn)].set_index("length_bin").reindex(labs)
            okk = d["mse_ratio"].notna().to_numpy()
            if okk.sum() < 3:
                continue
            sl = np.polyfit(centers[okk] / 100, d["mse_ratio"].to_numpy()[okk], 1,
                            w=np.sqrt(d["n"].to_numpy()[okk]))[0]
            print(f"  {mdl[-26:]:>26}  {mn:>5}: {sl:+.4f}")
    print(f"\nwrote {a.out_dir}/gain_by_length.csv and length_breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
