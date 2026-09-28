#!/usr/bin/env python3
"""Collect several fit_regressor.py runs into one results table.

Each run is an --out-dir from fit_regressor.py (it holds summary.csv and, when the
paired bootstrap ran, bootstrap_vs_reference.csv). Give each a short label:

    python summarize_results.py \\
        --run "ridge"=agn_ridge  --run "MLP"=agn_mlp \\
        --run "ridge + stats"=agn_ridge_stats  --run "MLP + stats"=agn_mlp_stats \\
        --out-dir results_tables

Writes to --out-dir:
  results_wide.csv     one row per model (latent, window parsed from the name): R^2 for
                       every run x target, and the bootstrap gain over that run's
                       control with its 95% interval and P(better).
  results_long.csv     the same numbers, one row per model x run x target (for plotting).
  results_summary.md   readable tables:
                         1. per run and target: the control, the best model, its gain
                            over the control [95% CI], and how many models beat the
                            control (interval entirely above zero);
                         2. the top models for each target in the --headline run;
                         3. R^2 by latent size (best window per latent) in that run.
  results_summary.tex  table 1 as a LaTeX tabular (with --latex).

Gains always compare a model with the summary-statistics control fitted with the SAME
head in the SAME run, which is what the bootstrap in fit_regressor.py measures.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = "CONTROL: summary stats only"
TARGETS = {"z": "z", "m": "log M_BH", "e": "log L/L_Edd"}
PAT_LATENT = re.compile(r"_K_01_(\d+)_(\d+)$")
PAT_BASE = re.compile(r"_base_\d+_(\d+)_K_baseline$")


def parse(name: str):
    m = PAT_LATENT.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m = PAT_BASE.search(name)
    if m:
        return "base", int(m.group(1))
    return "?", None


def short(name: str) -> str:
    lat, ws = parse(name)
    if name == CONTROL:
        return "control (stats only)"
    return f"latent {lat}, window {ws}" if ws else name


def load_run(label: str, d: Path, targets: list[str]) -> pd.DataFrame:
    s = pd.read_csv(d / "summary.csv")
    b_path = d / "bootstrap_vs_reference.csv"
    b = pd.read_csv(b_path) if b_path.is_file() else None
    if b is None:
        print(f"note: {b_path} not found; gains will be point estimates without intervals",
              file=sys.stderr)
    ctrl = s[s["model"] == CONTROL]
    rows = []
    for _, r in s.iterrows():
        lat, ws = parse(r["model"])
        for t in targets:
            if f"r2_{t}" not in s.columns:
                continue
            if r["model"] == CONTROL:
                lat = "-"
            row = {"run": label, "model": r["model"], "latent": lat,
                   "window": ws if r["model"] != CONTROL else -1, "target": t,
                   "r2": float(r[f"r2_{t}"]),
                   "rmse": float(r.get(f"rmse_{t}", np.nan))}
            if len(ctrl):
                row["r2_control"] = float(ctrl.iloc[0][f"r2_{t}"])
            if b is not None and r["model"] != CONTROL:
                br = b[b["model"] == r["model"]]
                if len(br) and f"delta_r2_{t}" in b.columns:
                    br = br.iloc[0]
                    row.update(delta=float(br[f"delta_r2_{t}"]),
                               delta_lo=float(br[f"delta_r2_{t}_lo"]),
                               delta_hi=float(br[f"delta_r2_{t}_hi"]),
                               p_better=float(br[f"p_better_{t}"]))
            if "delta" not in row and r["model"] != CONTROL and "r2_control" in row:
                row["delta"] = row["r2"] - row["r2_control"]
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_gain(r) -> str:
    if pd.isna(r.get("delta")):
        return ""
    if pd.notna(r.get("delta_lo")):
        return f"{r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
    return f"{r['delta']:+.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="a fit_regressor.py --out-dir, with a label (repeatable)")
    ap.add_argument("--targets", default="z,m,e")
    ap.add_argument("--headline", default=None,
                    help="label of the run for tables 2-3 (default: the last --run)")
    ap.add_argument("--top", type=int, default=5, help="models per target in table 2")
    ap.add_argument("--latex", action="store_true", help="also write table 1 as LaTeX")
    ap.add_argument("--out-dir", type=Path, default=Path("results_tables"))
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"--run needs LABEL=DIR, got {spec!r}", file=sys.stderr)
            return 1
        label, d = spec.split("=", 1)
        d = Path(d)
        if not (d / "summary.csv").is_file():
            print(f"no summary.csv in {d}", file=sys.stderr)
            return 1
        runs.append((label, d))
    labels = [l for l, _ in runs]
    headline = args.headline or labels[-1]
    if headline not in labels:
        print(f"--headline {headline!r} is not one of {labels}", file=sys.stderr)
        return 1

    long = pd.concat([load_run(l, d, targets) for l, d in runs], ignore_index=True)
    long["window"] = pd.array(long["window"], dtype="Int64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "results_long.csv", index=False)

    # ---- wide ------------------------------------------------------------------
    vals = ["r2", "delta", "delta_lo", "delta_hi", "p_better"]
    vals = [v for v in vals if v in long.columns]
    wide = long.pivot_table(index=["model", "latent", "window"], columns=["run", "target"],
                            values=vals, aggfunc="first")
    wide.columns = [f"{v}_{t}_{run}" for v, run, t in wide.columns]
    order = [f"{v}_{t}_{run}" for run in labels for t in targets for v in vals
             if f"{v}_{t}_{run}" in wide.columns]
    wide = wide[order].reset_index()
    wide["_lat"] = pd.to_numeric(wide["latent"], errors="coerce").fillna(1e9)
    wide = wide.sort_values(["_lat", "window"]).drop(columns="_lat")
    wide.to_csv(args.out_dir / "results_wide.csv", index=False)

    # ---- table 1: per run x target ----------------------------------------------
    t1 = []
    for run in labels:
        for t in targets:
            g = long[(long["run"] == run) & (long["target"] == t)]
            c = g[g["model"] == CONTROL]
            m = g[g["model"] != CONTROL]
            if m.empty:
                continue
            best = m.loc[m["r2"].idxmax()]
            n_better = int((m["delta_lo"] > 0).sum()) if "delta_lo" in m else np.nan
            t1.append({"run": run, "target": TARGETS.get(t, t),
                       "control R2": f"{c['r2'].iloc[0]:.3f}" if len(c) else "",
                       "best model": short(best["model"]),
                       "best R2": f"{best['r2']:.3f}",
                       "gain [95% CI]": fmt_gain(best),
                       "models beating control": (f"{n_better}/{len(m)}"
                                                  if not pd.isna(n_better) else "")})
    t1 = pd.DataFrame(t1)

    # ---- table 2: top models in the headline run ---------------------------------
    t2_parts = []
    for t in targets:
        g = long[(long["run"] == headline) & (long["target"] == t) & (long["model"] != CONTROL)]
        for rank, (_, r) in enumerate(g.sort_values("r2", ascending=False).head(args.top).iterrows(), 1):
            t2_parts.append({"target": TARGETS.get(t, t), "rank": rank, "model": short(r["model"]),
                             "R2": f"{r['r2']:.3f}", "gain [95% CI]": fmt_gain(r),
                             "P(better)": f"{r['p_better']:.3f}" if pd.notna(r.get("p_better")) else ""})
    t2 = pd.DataFrame(t2_parts)

    # ---- table 3: best window per latent, headline run ---------------------------
    h = long[(long["run"] == headline) & (long["model"] != CONTROL)]
    t3_rows = []
    lat_order = sorted([l for l in h["latent"].unique() if str(l).isdigit()], key=int) + \
        [l for l in h["latent"].unique() if not str(l).isdigit()]
    for lat in lat_order:
        row = {"latent": lat}
        for t in targets:
            g = h[(h["latent"] == lat) & (h["target"] == t)]
            if g.empty:
                continue
            b = g.loc[g["r2"].idxmax()]
            row[f"{TARGETS.get(t, t)} R2 (window)"] = f"{b['r2']:.3f} ({int(b['window'])})"
            row[f"{TARGETS.get(t, t)} mean over windows"] = f"{g['r2'].mean():.3f}"
        t3_rows.append(row)
    t3 = pd.DataFrame(t3_rows)

    md = ["# AGN regression results", "",
          f"Runs: {', '.join(f'**{l}** (`{d}`)' for l, d in runs)}.", "",
          "Gains are R^2 minus the summary-statistics control fitted with the same head in the "
          "same run, from the paired bootstrap over sources; a model 'beats the control' when "
          "the whole 95% interval is above zero.", "",
          "## 1. Best model per run and target", "", md_table(t1), "",
          f"## 2. Top {args.top} models per target: {headline}", "", md_table(t2), "",
          f"## 3. By latent size: {headline} (best window, and mean over windows)", "",
          md_table(t3), ""]
    (args.out_dir / "results_summary.md").write_text("\n".join(md))

    if args.latex:
        tex = ["\\begin{tabular}{llllll}", "\\hline",
               "Run & Target & Control $R^2$ & Best model & Best $R^2$ & Gain [95\\% CI] \\\\",
               "\\hline"]
        for _, r in t1.iterrows():
            cells = [r["run"], r["target"], r["control R2"], r["best model"], r["best R2"],
                     r["gain [95% CI]"]]
            cells = [str(c).replace("\\", "\\textbackslash{}").replace("_", "\\_")
                     .replace("%", "\\%").replace("&", "\\&") for c in cells]
            tex.append(" & ".join(cells) + " \\\\")
        tex += ["\\hline", "\\end{tabular}"]
        (args.out_dir / "results_summary.tex").write_text("\n".join(tex) + "\n")

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(t1.to_string(index=False))
    print(f"\nwrote {args.out_dir}/results_summary.md, results_wide.csv, results_long.csv"
          + (", results_summary.tex" if args.latex else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
