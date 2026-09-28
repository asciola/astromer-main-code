#!/usr/bin/env python3
"""True vs out-of-fold predicted AGN parameter, for one model in the sweep.

Reads the predictions.csv that fit_regressor.py writes (one row per source per
model, carrying <target>_true and <target>_pred) and draws three panels:

  a  true vs predicted, with the 1:1 line, the fitted line and quantile-binned means.
  b  residual (pred - true) vs TRUE.
  c  residual (pred - true) vs PREDICTED.

Panels b and c answer different questions, and only c is a diagnostic.

  b always slopes downward when R^2 < 1. A model that explains a fraction of the
    variance hedges toward the mean, so truly low values are over-predicted and
    truly high ones under-predicted. For a least-squares fit the slope of b is
    exactly (slope of a) - 1; it is drawn as a reference line. A steep b means a
    weak model, not a broken one.
  c should be FLAT at zero for well-calibrated out-of-fold predictions. A slope or
    curve here is a real, fixable problem (e.g. an MLP that over- or under-shrinks).

The binned means carry the figure. With ~170k sources the raw points are drawn as
a density (hexbin) rather than a scatter, which would be a solid blob.

--compare another predictions.csv (e.g. the MLP run vs the ridge run) overlays that
file's binned means on panels a and c for the same model name.

--index lc_index.csv additionally writes residual_diagnostics_<t>_ws<w>.png:
the residual against quantities the model might be missing - redshift, mean
r-band magnitude, number of r-band points, true log L_bol, and the z<0.7 L_bol
caveat flag. Structure there says what the embedding does not capture.

    python plot_true_vs_pred.py --pred agn_results/predictions.csv --list
    python plot_true_vs_pred.py --pred agn_results/predictions.csv --window 800 --model mhla_4
    python plot_true_vs_pred.py --pred agn_results_ridge/predictions.csv --window 800 \\
        --model mhla_4 --compare agn_results_mlp/predictions.csv \\
        --label ridge --compare-label MLP --index lc_index.csv
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
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, RULE = "#101A19", "#384645", "#5F6B6A", "#DBE1E0"
ACCENT, ACCENT2 = "#0E8C7E", "#A06A00"
# point density is context, not a series: a neutral single-hue ramp (light -> dark
# gray) so the teal / amber binned means stay the readable layer on top of it
DENSITY = LinearSegmentedColormap.from_list("density", ["#F1F3F3", "#C9D0CF", "#8E9998", "#4A5554"])
HEX_MIN_N = 3000        # below this, a scatter is still readable

TARGET_LABEL = {"m": r"$\log M_{\rm BH}$", "z": r"$z$", "e": r"$\log\,L/L_{\rm Edd}$"}


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def auto_pick(names: list[str], latent: int) -> str | None:
    """Best guess at the 'MHLA, latent=<latent>' model name, whatever the convention."""
    tok = re.compile(rf"(?<![0-9]){latent}(?![0-9])")
    scored = []
    for n in names:
        low = n.lower()
        s = 0
        if "mhla" in low or "latent" in low or "_ld" in low:
            s += 2
        if tok.search(low):
            s += 3
        if "standard" in low or "vanilla" in low or "base" in low:
            s -= 5
        scored.append((s, -len(n), n))
    scored.sort(reverse=True)
    return scored[0][2] if scored and scored[0][0] > 0 else None


def binned(x, v, bins):
    """Quantile bins of x: centres, mean of v, standard error of v, 16/84 pct of v."""
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    qs[-1] += 1e-9
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        s = idx == b
        if s.sum() < 3:
            continue
        vv = v[s]
        out.append((x[s].mean(), vv.mean(), vv.std(ddof=1) / np.sqrt(s.sum()),
                    np.percentile(vv, 16), np.percentile(vv, 84)))
    return [np.asarray(c) for c in zip(*out)]


def select(df, tcol, pcol, window, model_q, latent, allow_auto=True, quiet=False):
    sub = df
    if window is not None and "window_size" in sub.columns:
        at_ws = sub[sub["window_size"] == window]
        if at_ws.empty:
            avail = sorted(sub["window_size"].unique())
            raise SystemExit(f"no rows with window_size={window}; available: {avail}")
        sub = at_ws
    names = sorted(sub["model"].unique())
    if model_q:
        exact = [n for n in names if n == model_q]
        hits = exact or [n for n in names if model_q.lower() in n.lower()]
        if len(hits) != 1:
            msg = (f"--model {model_q!r} matched {len(hits)} models at window {window}."
                   "\navailable:\n  " + "\n  ".join(names))
            raise SystemExit(msg)
        model = hits[0]
    elif allow_auto:
        model = auto_pick(names, latent)
        if model is None:
            raise SystemExit(f"could not guess the latent-{latent} model. Pass --model. "
                             f"Available at window {window}:\n  " + "\n  ".join(names))
        if not quiet:
            print(f"auto-picked model {model!r} (override with --model)")
    else:
        return None, None
    d = sub[sub["model"] == model].copy()
    if "source_id" in d.columns and d["source_id"].duplicated().any():
        n0 = len(d)
        d = d.groupby("source_id", as_index=False)[[tcol, pcol]].mean()
        print(f"note: averaged {n0:,} rows over duplicate source_id -> {len(d):,} sources "
              f"(multiple seeds stored)")
    return model, d


def arrays(d, tcol, pcol):
    y = d[tcol].to_numpy(float)
    p = d[pcol].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < len(y):
        print(f"note: dropped {len(y) - int(ok.sum())} non-finite row(s)")
    return y[ok], p[ok], d.loc[ok, "source_id"].astype(str).to_numpy() if "source_id" in d else None


def style_axes(ax):
    ax.grid(True, color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def density(ax, x, y, extent=None):
    """Scatter for small n, log-scaled hexbin for large n."""
    if len(x) < HEX_MIN_N:
        ax.scatter(x, y, s=11, c="#8E9998", alpha=0.35, lw=0, zorder=2)
        return None
    return ax.hexbin(x, y, gridsize=70, bins="log", cmap=DENSITY, mincnt=1,
                     linewidths=0, extent=extent, zorder=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--target", default="m", help="target key, e.g. m / z / e (default m)")
    ap.add_argument("--model", default=None, help="model name, or a substring of it")
    ap.add_argument("--window", type=int, default=800, help="window_size to select (default 800)")
    ap.add_argument("--latent", type=int, default=4, help="latent dim to auto-pick (default 4)")
    ap.add_argument("--bins", type=int, default=12, help="quantile bins for the binned means")
    ap.add_argument("--compare", type=Path, default=None,
                    help="second predictions.csv (e.g. the MLP run) to overlay")
    ap.add_argument("--label", default=None, help="legend name for --pred (default: its head)")
    ap.add_argument("--compare-label", default=None, help="legend name for --compare")
    ap.add_argument("--index", type=Path, default=None,
                    help="lc_index.csv: also write the residual-vs-covariates figure")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="print available models and exit")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if not args.pred.is_file():
        print(f"not found: {args.pred}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.pred, dtype={"source_id": str})

    tcol, pcol = f"{args.target}_true", f"{args.target}_pred"
    if tcol not in df.columns or pcol not in df.columns:
        tgts = sorted({c[:-5] for c in df.columns if c.endswith("_true")})
        print(f"no columns {tcol}/{pcol}. targets present: {tgts}", file=sys.stderr)
        return 1
    if "model" not in df.columns:
        print(f"no 'model' column; saw {list(df.columns)}", file=sys.stderr)
        return 1

    if args.list:
        print(f"{args.pred}: {len(df):,} rows\n")
        g = (df.groupby(["model"] + (["window_size"] if "window_size" in df else []))
               .size().rename("rows").reset_index())
        for _, r in g.iterrows():
            ws = f"  ws={int(r['window_size']):>5}" if "window_size" in g else ""
            print(f"  {r['model']:<44}{ws}  {int(r['rows']):>6,} rows")
        return 0

    try:
        model, d = select(df, tcol, pcol, args.window, args.model, args.latent)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    y, p, sids = arrays(d, tcol, pcol)
    if y.size < 10:
        print(f"only {y.size} usable rows", file=sys.stderr)
        return 1
    head = str(df["head"].iloc[0]) if "head" in df.columns else "ridge"
    lab_a = args.label or head

    cmp = None
    if args.compare is not None:
        dfc = pd.read_csv(args.compare, dtype={"source_id": str})
        if tcol not in dfc.columns:
            print(f"--compare file lacks {tcol}", file=sys.stderr)
            return 1
        try:
            _, dc = select(dfc, tcol, pcol, args.window, model, args.latent, quiet=True)
        except SystemExit as e:
            print(f"--compare: {e}", file=sys.stderr)
            return 1
        yc, pc, _ = arrays(dc, tcol, pcol)
        head_c = str(dfc["head"].iloc[0]) if "head" in dfc.columns else "compare"
        cmp = {"y": yc, "p": pc, "label": args.compare_label or head_c}

    R2 = r2(y, p)
    rho = float(np.corrcoef(y, p)[0, 1])
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    slope, icept = np.polyfit(y, p, 1)
    res = p - y
    s_rp, i_rp = np.polyfit(p, res, 1)          # residual vs predicted: should be ~0

    print(f"\nmodel      {model}")
    print(f"window     {args.window}")
    print(f"target     {args.target}   n = {y.size:,}   head = {lab_a}")
    print(f"R^2        {R2:+.4f}")
    print(f"pearson r  {rho:+.4f}   (r^2 = {rho**2:.4f})")
    print(f"RMSE       {rmse:.4f}   sd(true) = {y.std(ddof=1):.4f}")
    print(f"slope      {slope:.3f}  (1.0 = unbiased; below that is shrinkage)")
    print(f"residual-vs-true slope      {slope - 1:+.3f}  (= slope - 1, expected)")
    print(f"residual-vs-predicted slope {s_rp:+.3f}  (should be ~0; this is the diagnostic)")
    if cmp:
        print(f"compare ({cmp['label']}): R^2 {r2(cmp['y'], cmp['p']):+.4f}, "
              f"slope {np.polyfit(cmp['y'], cmp['p'], 1)[0]:.3f}")

    bx, bp, bse, _, _ = binned(y, p, args.bins)
    rx, rm, rse, _, _ = binned(p, res, args.bins)
    print(f"\nbinned means ({len(bx)} bins): true {bx.min():.2f}->{bx.max():.2f}, "
          f"pred {bp.min():.2f}->{bp.max():.2f}  "
          f"(monotonic: {bool(np.all(np.diff(bp) > 0))})")

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.0, 5.0),
                                     gridspec_kw={"wspace": 0.30})
    tl = TARGET_LABEL.get(args.target, args.target)
    for ax in (a1, a2, a3):
        ax.set_box_aspect(1)          # square panels, aligned titles; a keeps 1:1 limits

    # ---- a: true vs predicted -------------------------------------------------
    lo = min(np.percentile(y, 0.2), np.percentile(p, 0.2)) - 0.15
    hi = max(np.percentile(y, 99.8), np.percentile(p, 99.8)) + 0.15
    hb = density(a1, y, p, extent=(lo, hi, lo, hi))
    a1.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3, label="1:1")
    a1.plot([lo, hi], [slope * lo + icept, slope * hi + icept],
            color=INK2, lw=1.4, zorder=4, label=f"fit (slope {slope:.2f})")
    a1.errorbar(bx, bp, yerr=bse, fmt="o", ms=6.5, color=ACCENT, mec="white",
                mew=1.4, ecolor=ACCENT, elinewidth=1.4, capsize=0, zorder=5,
                label=f"binned means, {lab_a}")
    if cmp:
        cbx, cbp, cbse, _, _ = binned(cmp["y"], cmp["p"], args.bins)
        a1.errorbar(cbx, cbp, yerr=cbse, fmt="s", ms=6.0, color=ACCENT2, mec="white",
                    mew=1.4, ecolor=ACCENT2, elinewidth=1.4, capsize=0, zorder=5,
                    label=f"binned means, {cmp['label']}")
    a1.set_xlim(lo, hi); a1.set_ylim(lo, hi)
    a1.set_xlabel(f"true {tl}")
    a1.set_ylabel(f"predicted {tl}  (out-of-fold)")
    a1.set_title("a   true vs predicted", loc="left", fontsize=9.5, color=INK)
    style_axes(a1)
    a1.legend(frameon=False, loc="upper left", fontsize=7.8, handletextpad=0.6)
    stats = f"$R^2$ = {R2:.3f}\n$r$ = {rho:.3f}\nRMSE = {rmse:.3f}\nn = {y.size:,}"
    if cmp:
        stats += f"\n{cmp['label']} $R^2$ = {r2(cmp['y'], cmp['p']):.3f}"
    a1.text(0.97, 0.04, stats, transform=a1.transAxes, ha="right", va="bottom",
            fontsize=8.2, color=INK2, linespacing=1.5)

    # ---- b: residual vs true --------------------------------------------------
    rlo, rhi = np.percentile(res, [0.2, 99.8])
    pad = 0.1 * (rhi - rlo)
    density(a2, y, res, extent=(lo, hi, rlo - pad, rhi + pad))
    a2.axhline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
    xx = np.array([lo, hi])
    a2.plot(xx, (slope - 1) * xx + icept, color=INK2, lw=1.4, zorder=4,
            label=f"expected: slope $-$ 1 = {slope - 1:.2f}")
    tb_x, tb_m, tb_se, _, _ = binned(y, res, args.bins)
    a2.errorbar(tb_x, tb_m, yerr=tb_se, fmt="o", ms=6.0, color=ACCENT, mec="white",
                mew=1.4, ecolor=ACCENT, elinewidth=1.4, capsize=0, zorder=5,
                label="binned mean residual")
    a2.set_xlim(lo, hi); a2.set_ylim(rlo - pad, rhi + pad)
    a2.set_xlabel(f"true {tl}")
    a2.set_ylabel("residual  (predicted $-$ true)")
    a2.set_title("b   residual vs true  (slopes by construction)", loc="left",
                 fontsize=9.5, color=INK)
    style_axes(a2)
    a2.legend(frameon=False, loc="upper right", fontsize=7.8)

    # ---- c: residual vs predicted ---------------------------------------------
    plo, phi = np.percentile(p, [0.2, 99.8])
    ppad = 0.05 * (phi - plo)
    density(a3, p, res, extent=(plo - ppad, phi + ppad, rlo - pad, rhi + pad))
    a3.axhline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3, label="zero (calibrated)")
    a3.errorbar(rx, rm, yerr=rse, fmt="o-", ms=6.0, lw=1.4, color=ACCENT, mec="white",
                mew=1.4, ecolor=ACCENT, elinewidth=1.4, capsize=0, zorder=5,
                label=f"binned mean, {lab_a}  (slope {s_rp:+.3f})")
    if cmp:
        cres = cmp["p"] - cmp["y"]
        crx, crm, crse, _, _ = binned(cmp["p"], cres, args.bins)
        cs = np.polyfit(cmp["p"], cres, 1)[0]
        a3.errorbar(crx, crm, yerr=crse, fmt="s-", ms=5.5, lw=1.4, color=ACCENT2,
                    mec="white", mew=1.4, ecolor=ACCENT2, elinewidth=1.4, capsize=0,
                    zorder=5, label=f"binned mean, {cmp['label']}  (slope {cs:+.3f})")
    a3.set_xlim(plo - ppad, phi + ppad); a3.set_ylim(rlo - pad, rhi + pad)
    a3.set_xlabel(f"predicted {tl}")
    a3.set_ylabel("residual  (predicted $-$ true)")
    a3.set_title("c   residual vs predicted  (should be flat)", loc="left",
                 fontsize=9.5, color=INK)
    style_axes(a3)
    a3.legend(frameon=False, loc="upper right", fontsize=7.8)

    if hb is not None:
        cax = fig.add_axes([0.915, 0.18, 0.006, 0.64])
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("sources per cell", color=INK2, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED)

    ttl = args.title or f"{model}  ·  window {args.window}"
    fig.suptitle(ttl, x=0.5, y=1.0, fontsize=11, color=INK)
    fig.text(0.5, -0.04,
             "A model that explains part of the variance hedges toward the mean, so the residual "
             "against the TRUE value (b) slopes by exactly the fit slope minus 1.\nThat is expected. "
             "The diagnostic is c: out-of-fold residuals against the PREDICTED value should sit "
             "flat on zero; a slope there means miscalibration.",
             ha="center", va="top", fontsize=7.8, color=MUTED, linespacing=1.6)

    out = args.out or Path(f"true_vs_pred_{args.target}_ws{args.window}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}")

    if args.index is not None:
        out2 = out.with_name(out.name.replace("true_vs_pred", "residual_diagnostics", 1)
                             if out.name.startswith("true_vs_pred")
                             else "residual_diagnostics_" + out.name)
        covariate_figure(args, model, y, p, sids, out2, cmp_df=None)
    return 0


def covariate_figure(args, model, y, p, sids, out, cmp_df=None):
    """Residual vs quantities the embedding may not capture."""
    idx = pd.read_csv(args.index, dtype={"source_id": str})
    per_src = idx.drop_duplicates("source_id").set_index("source_id")
    r_band = idx[idx["band_name"] == "r"].drop_duplicates("source_id").set_index("source_id")
    res = p - y
    covs = []
    if args.target != "z" and "Z_FIT" in per_src:
        covs.append(("redshift  $z$", "redshift z", per_src["Z_FIT"].reindex(sids).to_numpy(float)))
    if args.target != "m" and "LOGMBH" in per_src:
        m = per_src["LOGMBH"].reindex(sids).to_numpy(float)
        covs.append((r"true $\log M_{\rm BH}$", "true log M_BH", np.where(m > 0, m, np.nan)))
    if "mag_mean" in r_band:
        covs.append(("mean r magnitude", "mean r magnitude", r_band["mag_mean"].reindex(sids).to_numpy(float)))
    if "n_points" in r_band:
        covs.append(("r-band points", "r-band points", r_band["n_points"].reindex(sids).to_numpy(float)))
    if "LOGLBOL" in per_src:
        lb = per_src["LOGLBOL"].reindex(sids).to_numpy(float)
        covs.append((r"true $\log L_{\rm bol}$", "true log L_bol", np.where(lb > 0, lb, np.nan)))
    flag = (per_src["lbol_lowz_caveat"].reindex(sids).astype(str).str.lower() == "true"
            ).to_numpy() if "lbol_lowz_caveat" in per_src else None

    n = len(covs) + (1 if flag is not None else 0)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow),
                             gridspec_kw={"wspace": 0.28, "hspace": 0.45}, squeeze=False)
    axes = axes.ravel()
    rlo, rhi = np.percentile(res, [1, 99])
    print(f"\nresidual trends ({args.target}):")
    for ax, (name, plain, x) in zip(axes, covs):
        ok = np.isfinite(x)
        xs, rs = x[ok], res[ok]
        cx, cm, cse, c16, c84 = binned(xs, rs, args.bins)
        ax.fill_between(cx, c16, c84, color=ACCENT, alpha=0.14, lw=0,
                        label="16–84% of residuals")
        ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
        ax.errorbar(cx, cm, yerr=cse, fmt="o-", ms=5.5, lw=1.4, color=ACCENT, mec="white",
                    mew=1.2, ecolor=ACCENT, elinewidth=1.3, capsize=0, label="binned mean")
        span = float(cm.max() - cm.min())
        ax.set_xlabel(name)
        ax.set_ylabel("residual (pred $-$ true)")
        ax.set_ylim(rlo, rhi)
        ax.set_title(f"range of binned means {span:.3f}", loc="left", fontsize=8.5, color=INK2)
        style_axes(ax)
        print(f"  {plain:<28} binned-mean range {span:.3f}   (n={ok.sum():,})")
    k = len(covs)
    if flag is not None:
        ax = axes[k]
        groups = [("z ≥ 0.7", ~flag), ("z < 0.7\n(L_bol caveat)", flag)]
        for j, (nm, s) in enumerate(groups):
            if s.sum() < 3:
                continue
            v = res[s]
            lo16, hi84 = np.percentile(v, [16, 84])
            ax.plot([j, j], [lo16, hi84], color=ACCENT, lw=8, alpha=0.18, solid_capstyle="butt")
            ax.errorbar([j], [v.mean()], yerr=[v.std(ddof=1) / np.sqrt(len(v))], fmt="o",
                        ms=7, color=ACCENT, mec="white", mew=1.2, ecolor=ACCENT, capsize=0)
            ax.text(j + 0.16, v.mean(), f"{v.mean():+.3f}\nn={len(v):,}", va="center",
                    fontsize=7.5, color=INK2)
        ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
        ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups])
        ax.set_xlim(-0.5, 1.9); ax.set_ylim(rlo, rhi)
        ax.set_ylabel("residual (pred $-$ true)")
        ax.set_title("mean residual by L_bol caveat", loc="left", fontsize=8.5, color=INK2)
        style_axes(ax)
        print(f"  lowz caveat: mean residual {res[flag].mean():+.3f} (n={flag.sum():,}) vs "
              f"{res[~flag].mean():+.3f}")
        k += 1
    for ax in axes[k:]:
        ax.axis("off")
    if len(covs):
        axes[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    tl = TARGET_LABEL.get(args.target, args.target)
    fig.suptitle(f"{tl} residuals vs other quantities  ·  {model}  ·  window {args.window}",
                 y=1.0, fontsize=11, color=INK)
    fig.text(0.5, -0.02,
             "Flat binned means = the model's errors do not depend on this quantity. A trend means "
             "part of the signal (or a bias) tracks it and the embedding is not capturing it.",
             ha="center", va="top", fontsize=7.8, color=MUTED)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
