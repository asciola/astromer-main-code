#!/usr/bin/env python3
"""Are two feature sets carrying the SAME signal, or complementary signal?

Both the embeddings and the 18 summary statistics reach R^2 ~ 0.124 on log M_BH, and
combining them into one ridge gains nothing. Two different things produce that pattern:

  redundant     -- they encode the same information. Out-of-fold predictions correlate
                   tightly, and a stack of the two does no better than either alone.
  complementary -- they encode different information, but one ridge with a single alpha
                   could not exploit it (256 columns swamp 18). Predictions correlate only
                   loosely, and a two-predictor stack beats both.

This reads the out-of-fold predictions that fit_regressor.py already wrote and decides,
without refitting anything on features. For each pair it reports R^2 of each model, the
correlation between their predictions, how much of one's RESIDUAL the other explains, and
a cross-validated stack of the two.

    python oof_complementarity.py agn_results/predictions.csv --target m
    python oof_complementarity.py agn_emb_only/predictions.csv agn_results/predictions.csv \
        --target m --label emb_only combined

Column layout is discovered, not assumed: it handles a long table (one row per
source x model) and a wide one (<model>_<target>_pred columns), and looks for
'<target>_true' / '<target>_pred' pairs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ID_CANDIDATES = ("source_id", "lcid", "ID", "id", "objectid", "newID", "source",
                 "oid", "obj_id", "sourceid")
MODEL_CANDIDATES = ("model", "exp_name", "exp", "embedding", "emb", "name", "run")


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows, list(rows[0])


def pick(cols: list[str], candidates) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def detect_columns(rows: list[dict], cols: list[str], id_override: str | None,
                   model_override: str | None) -> tuple[str | None, str | None]:
    """Find the source-id and model columns by TESTING them, not by trusting names.

    Naming varies between runs, and guessing wrong is silently catastrophic: a wrong id
    column collapses every row onto one key, which yields 1 "source", zero variance and
    R2 = -inf. So a candidate id column has to actually behave like one -- unique within a
    model group, and the same value set across groups.
    """
    if id_override and id_override not in cols:
        raise SystemExit(f"--id-column '{id_override}' not in {cols}")
    if model_override and model_override not in cols:
        raise SystemExit(f"--model-column '{model_override}' not in {cols}")

    mdc = model_override or pick(cols, MODEL_CANDIDATES)
    if mdc is None:
        # a column with few distinct values, each repeated many times, looks like a model key
        for c in cols:
            vals = [r[c] for r in rows]
            u = set(vals)
            if 1 < len(u) <= max(4, len(rows) // 20) and len(vals) % len(u) == 0:
                mdc = c
                break

    groups = {}
    for r in rows:
        groups.setdefault(r[mdc] if mdc else "_all", []).append(r)
    big = max(groups.values(), key=len)

    def behaves_like_id(c: str) -> bool:
        v = [r[c] for r in big]
        if len(set(v)) != len(v):
            return False                      # repeats inside one model group
        sets = [{r[c] for r in g} for g in groups.values() if len(g) == len(big)]
        return all(x == sets[0] for x in sets)  # same sources for every model

    if id_override:
        if not behaves_like_id(id_override):
            raise SystemExit(f"--id-column '{id_override}' is not unique within a model group")
        return id_override, mdc

    named = pick(cols, ID_CANDIDATES)
    if named and behaves_like_id(named):
        return named, mdc
    for c in cols:                            # any column that passes the test
        if c != mdc and behaves_like_id(c):
            return c, mdc
    return None, mdc                          # caller falls back to row order


def collect(path: Path, target: str, tag: str, args) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """-> ({model_name: predictions}, truth, ids). Long and wide layouts handled."""
    rows, cols = read_csv(path)
    tcol = next((c for c in cols if c.lower() == f"{target}_true".lower()), None)
    pcol = next((c for c in cols if c.lower() == f"{target}_pred".lower()), None)

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    if tcol and pcol:
        idc, mdc = detect_columns(rows, cols, args.id_column, args.model_column)
        print(f"  columns: id={idc or '(row order)'}  model={mdc or '(single)'}  "
              f"truth={tcol}  pred={pcol}")
        if idc is None:
            # positional fallback: every model must list the same sources in the same order
            groups = {}
            for r in rows:
                groups.setdefault(r[mdc] if mdc else "_all", []).append(r)
            sizes = {len(g) for g in groups.values()}
            if len(sizes) != 1:
                raise SystemExit(
                    f"{path}: no usable id column, and the {len(groups)} model groups have "
                    f"different row counts {sorted(sizes)} -- cannot align them by position.\n"
                    f"  columns seen: {cols}\n  pass --id-column <name>.")
            n = sizes.pop()
            print(f"  !! no id column found; aligning {len(groups)} model(s) by row order "
                  f"({n} rows each). Pass --id-column if that is wrong.")
            ids = [str(i) for i in range(n)]
            first = next(iter(groups.values()))
            y = np.array([f(r[tcol]) for r in first])
            out = {(f"{tag}:{k}" if tag else k): np.array([f(r[pcol]) for r in g])
                   for k, g in groups.items()}
            return out, y, ids

        by, truth = {}, {}
        for r in rows:
            sid = r[idc]
            by.setdefault(r[mdc] if mdc else "_all", {})[sid] = f(r[pcol])
            truth[sid] = f(r[tcol])
        ids = sorted(truth)
        if len(ids) < 10:
            raise SystemExit(
                f"{path}: only {len(ids)} distinct source id(s) in column '{idc}' -- that is "
                f"not a per-source table.\n  columns seen: {cols}\n"
                f"  pass --id-column <name> naming the source identifier.")
        y = np.array([truth[i] for i in ids])
        out = {}
        for k, d in by.items():
            if len(d) >= 0.5 * len(ids):
                out[f"{tag}:{k}" if tag else k] = np.array([d.get(i, np.nan) for i in ids])
        return out, y, ids

    # wide: <something>_<target>_pred  +  one <target>_true
    truth_col = next((c for c in cols if c.lower().endswith(f"{target}_true".lower())), None)
    pred_cols = [c for c in cols if c.lower().endswith(f"{target}_pred".lower()) and c != truth_col]
    if not truth_col or not pred_cols:
        raise SystemExit(
            f"{path}: could not find '{target}_true'/'{target}_pred' columns.\n"
            f"  columns seen: {cols[:14]}{' ...' if len(cols) > 14 else ''}\n"
            f"  pass a different --target, or tell me the column names.")
    idc = args.id_column or pick(cols, ID_CANDIDATES)
    ids = [r[idc] if idc else str(i) for i, r in enumerate(rows)]
    y = np.array([f(r[truth_col]) for r in rows])
    out = {}
    for c in pred_cols:
        nm = c[: -len(f"{target}_pred")].rstrip("_") or c
        out[f"{tag}:{nm}" if tag else nm] = np.array([f(r[c]) for r in rows])
    return out, y, ids


def r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0:                    # constant truth -> R2 undefined, not -inf
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def stack_cv(y: np.ndarray, a: np.ndarray, b: np.ndarray, folds: int = 5,
             seed: int = 0) -> float:
    """CV R^2 of least-squares on [a, b] -- honest about the 3 fitted parameters."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    pred = np.full(len(y), np.nan)
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        X = np.column_stack([np.ones(len(tr)), a[tr], b[tr]])
        coef, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        pred[te] = coef[0] + coef[1] * a[te] + coef[2] * b[te]
    return r2(y, pred)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", type=Path, help="predictions.csv file(s)")
    ap.add_argument("--label", nargs="*", default=None, help="tag per file, in order")
    ap.add_argument("--target", default="m", help="target prefix: m, z, e (default m)")
    ap.add_argument("--pairs", default="vs-control",
                    choices=("vs-control", "all", "best-vs-control"),
                    help="which pairs to report (default: everything against the control)")
    ap.add_argument("--control-match", default="control",
                    help="substring identifying the control model (default 'control')")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=8, help="max pairs to print")
    ap.add_argument("--id-column", default=None,
                    help="source identifier column, if auto-detection picks wrong")
    ap.add_argument("--model-column", default=None, help="model/experiment name column")
    ap.add_argument("--list-models", action="store_true",
                    help="just list the models found, with their R2, and stop")
    args = ap.parse_args()

    labels = args.label or [""] * len(args.predictions)
    while len(labels) < len(args.predictions):
        labels.append(args.predictions[len(labels)].parent.name)

    preds: dict[str, np.ndarray] = {}
    y = None
    for p, tag in zip(args.predictions, labels):
        d, yy, ids = collect(p, args.target, tag, args)
        print(f"{p}: {len(d)} model(s), {len(yy):,} sources")
        if y is None:
            y, n = yy, len(yy)
        elif len(yy) != len(y):
            print(f"  !! {len(yy)} sources here vs {len(y)} in the first file -- "
                  f"cannot compare across files with different source sets", file=sys.stderr)
            return 1
        preds.update(d)

    ok = np.isfinite(y)
    for v in preds.values():
        ok &= np.isfinite(v)
    dropped = int((~ok).sum())
    y = y[ok]
    preds = {k: v[ok] for k, v in preds.items()}
    print(f"\ntarget '{args.target}': {len(y):,} sources with complete predictions"
          f"{f' ({dropped} dropped as non-finite)' if dropped else ''}")

    scores = {k: r2(y, v) for k, v in preds.items()}
    ctrl = [k for k in preds if args.control_match.lower() in k.lower()]
    if args.list_models:
        print(f"\n{len(preds)} model(s) in this file:")
        for k in sorted(preds, key=lambda k: -(scores[k] if np.isfinite(scores[k]) else -9)):
            print(f"   {scores[k]:+.3f}  {k}")
        return 0
    if not ctrl and args.pairs != "all":
        print(f"\nNo model matching --control-match '{args.control_match}'. "
              f"{len(preds)} model(s) found, best first:")
        for k in sorted(preds, key=lambda k: -(scores[k] if np.isfinite(scores[k]) else -9))[:12]:
            print(f"   {scores[k]:+.3f}  {k}")
        print("\nThe key comparison needs the CONTROL's out-of-fold predictions in this file.")
        print("If fit_regressor.py does not write them, they have to be added before the")
        print("embeddings-vs-statistics question can be answered from predictions.csv.")
        print("Otherwise: --control-match <substring>, or --pairs all to compare models to")
        print("each other (n(n-1)/2 pairs -- use --top).")
        return 2
    control = ctrl[0] if ctrl else None
    if control:
        print(f"control: {control}   R2={scores[control]:+.3f}")

    if args.pairs == "all":
        pairs = list(itertools.combinations(sorted(preds, key=lambda k: -scores[k]), 2))
    elif args.pairs == "best-vs-control":
        best = max((k for k in preds if k != control), key=lambda k: scores[k])
        pairs = [(best, control)]
    else:
        pairs = [(k, control) for k in sorted(preds, key=lambda k: -scores[k]) if k != control]

    print(f"\n{'='*94}")
    print("Pairwise: do these two carry the same signal?")
    print(f"{'='*94}")
    hdr = (f"{'model A':<34}{'R2 A':>7}{'R2 B':>7}{'r(pA,pB)':>10}"
           f"{'r(resA,pB)':>12}{'stack R2':>10}{'gain':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    verdicts = []
    for a, b in pairs[: args.top]:
        pa, pb = preds[a], preds[b]
        ra, rb = scores[a], scores[b]
        r_pp = float(np.corrcoef(pa, pb)[0, 1])
        res_a = y - pa
        r_rp = float(np.corrcoef(res_a, pb)[0, 1])
        st = stack_cv(y, pa, pb, args.folds)
        gain = st - max(ra, rb)
        print(f"{a[:33]:<34}{ra:>+7.3f}{rb:>+7.3f}{r_pp:>10.3f}"
              f"{r_rp:>12.3f}{st:>+10.3f}{gain:>+7.3f}")
        verdicts.append((a, b, r_pp, gain))
    if len(pairs) > args.top:
        print(f"... ({len(pairs) - args.top} more pairs; raise --top)")

    print(f"\n{'='*94}")
    print("How to read it")
    print(f"{'='*94}")
    print("  r(pA,pB)    how alike the two prediction vectors are. >0.9 = the same signal;")
    print("              <0.7 with similar R2 = they disagree per-source, so there is")
    print("              something to combine.")
    print("  r(resA,pB)  whether B explains what A gets wrong. Near 0 = nothing left to add.")
    print("  stack R2    cross-validated least squares on the two predictions alone.")
    print("  gain        stack R2 minus the better of the two. This is the honest ceiling on")
    print("              what any smarter feature combination could buy you.")
    if verdicts:
        g = max(v[3] for v in verdicts)
        rr = min(v[2] for v in verdicts)
        print()
        # Correlation is decided FIRST. Two noisy estimates of the SAME signal still gain
        # from being averaged -- that is variance reduction, not extra information -- so a
        # positive stack gain at r>0.9 must not be read as complementarity.
        if rr > 0.9:
            print(f"  -> VERDICT: redundant. Predictions correlate at r={rr:.2f}: both feature")
            print("     sets are measuring one thing.")
            if g > 0.015:
                print(f"     The stack still gains {g:+.3f}, but at this correlation that is noise")
                print("     averaging over two estimates of the same quantity, not new signal.")
                print("     Averaging the two predictions is a free win; new features are not.")
            else:
                print(f"     Stack gain is only {g:+.3f}. Nothing to recover.")
        elif g > 0.015:
            print(f"  -> VERDICT: complementary. Predictions correlate at only r={rr:.2f} and the")
            print(f"     stack gains {g:+.3f}, so the single-alpha ridge was the limitation rather")
            print("     than the representation. Standardise per block or fit separate alphas.")
        else:
            print(f"  -> VERDICT: mostly redundant. Stack gain {g:+.3f} is within noise for ~950")
            print(f"     sources, though r={rr:.2f} is not high enough to call them identical.")
            print("     Worth a paired bootstrap on the stack before concluding either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
