#!/usr/bin/env python3
"""Check crossmatched labels (crossmatch_labels.py output) against an existing label file.

Every source in the old file that has a light curve should appear in the new file with
the same values. This reports, for the old sources:

  - how many have a light curve (are in --ids), and of those how many the crossmatch found;
  - which ones it missed, with their RA offset from the PS1-ID position, so you can tell a
    tolerance problem from a source that is simply absent from the catalog;
  - per-column agreement, and the rows that disagree.

IDs are read as strings throughout: 18-digit IDs do not survive a float64 round trip, and
pandas turns an ID column into float64 as soon as the file has one blank row.

    python check_labels.py --old comb.csv --new labels.csv --ids qzo_list.txt
    python check_labels.py --old comb.csv --new labels.csv --ids qzo_list.txt --out mismatches.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COLS = ["RA", "DEC", "Z_FIT", "Z_SYS_ERR", "LOGLBOL", "LOGLBOL_ERR", "LOGMBH", "LOGMBH_ERR"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    idc = df.columns[0]                       # 'id' (or 'd', as in the sample)
    df = df.rename(columns={idc: "id"})
    df["id"] = df["id"].str.strip()
    df = df[df["id"].str.fullmatch(r"\d+", na=False)]   # drops blank / junk rows
    for c in VALUE_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    dup = df["id"].duplicated().sum()
    if dup:
        print(f"warning: {path} has {dup} duplicated IDs; keeping the first", file=sys.stderr)
        df = df.drop_duplicates("id")
    return df.set_index("id")


def load_ids(src: Path) -> set[str]:
    names = (e.name for e in os.scandir(src)) if src.is_dir() else (l.strip() for l in open(src))
    out = set()
    for n in names:
        n = os.path.basename(n)
        s = n[:-4] if n.lower().endswith(".csv") else n
        if s.isdigit():
            out.add(s)
    return out


def ra_offset_arcsec(ids: pd.Index, ra: pd.Series, dec: pd.Series) -> np.ndarray:
    n = np.array([int(i) for i in ids], dtype=np.int64)
    ra_id = ((n // 10**5) % 10**8) / 1e5
    d = (ra_id - ra.to_numpy() + 180.0) % 360.0 - 180.0
    return np.abs(d) * np.cos(np.radians(dec.to_numpy())) * 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True, help="existing label file (e.g. comb.csv)")
    ap.add_argument("--new", type=Path, required=True, help="crossmatch_labels.py output")
    ap.add_argument("--ids", type=Path, default=None,
                    help="light-curve list or directory; restricts the check to old sources "
                         "that have a light curve")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance (default 1e-6)")
    ap.add_argument("--out", type=Path, default=None, help="write disagreeing rows here")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    print(f"old labels: {len(old):,}    new labels: {len(new):,}")

    pool = old.index
    if args.ids is not None:
        have = load_ids(args.ids)
        pool = old.index[old.index.isin(have)]
        print(f"old sources with a light curve in --ids: {len(pool):,} of {len(old):,}")

    found = pool[pool.isin(new.index)]
    missed = pool[~pool.isin(new.index)]
    print(f"found by the crossmatch: {len(found):,}    missed: {len(missed):,}")
    if len(missed):
        m = old.loc[missed]
        off = ra_offset_arcsec(missed, m["RA"], m["DEC"])
        print("  missed sources (RA offset between PS1 ID and old RA; >1\" means raise --ra-tol):")
        for i, o in sorted(zip(missed, off), key=lambda t: -t[1])[:20]:
            print(f"    {i}  dRA={o:.2f}\"")
        if len(missed) > 20:
            print(f"    ... and {len(missed) - 20} more")

    if not len(found):
        return 1
    a, b = old.loc[found], new.loc[found]
    bad_any = np.zeros(len(found), dtype=bool)
    print(f"\nper-column agreement on {len(found):,} sources (rtol={args.rtol:g}):")
    for c in VALUE_COLS:
        if c not in a or c not in b:
            print(f"  {c:12s} not in both files")
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        same = np.isclose(x, y, rtol=args.rtol, atol=0.0) | (np.isnan(x) & np.isnan(y))
        bad_any |= ~same
        worst = np.nanmax(np.abs(x - y)) if (~same).any() else 0.0
        print(f"  {c:12s} {same.sum():6,} agree   {(~same).sum():5,} differ   max |diff| {worst:.3g}")

    n_bad = int(bad_any.sum())
    print(f"\nsources with any disagreement: {n_bad:,}")
    if n_bad:
        cols = [c for c in VALUE_COLS if c in a and c in b]
        rep = a.loc[bad_any, cols].add_suffix("_old").join(b.loc[bad_any, cols].add_suffix("_new"))
        rep = rep[[f"{c}_{s}" for c in cols for s in ("old", "new")]]
        if args.out:
            rep.to_csv(args.out)
            print(f"wrote {args.out}")
        else:
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(rep.head(10))
    ok = not len(missed) and not n_bad
    print("\nPASS" if ok else "\nsee above")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
