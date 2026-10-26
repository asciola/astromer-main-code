#!/usr/bin/env python3
"""Build a quasar-label table for the ZTF light curves by crossmatching to Wu & Shen (2022).

The light-curve files are named by Pan-STARRS1 objID (<objID>.csv), and a PS1 objID
encodes the source position, so no position table is needed:

    objID = zone * 10**13  +  floor(RA * 1e5) * 10**5  +  <5 digits>
    zone  = floor((Dec + 90) / 0.0083333)

RA comes back to ~1e-5 deg. Dec comes back only to the 30"-tall zone the source sits in
(the last 5 digits do not decode to Dec). That is still enough: a catalog quasar is
accepted if its RA agrees to within --ra-tol arcsec and its Dec falls inside the zone
(padded by --dec-pad arcsec). At DR16Q densities (~75 per deg^2) the chance that an
unrelated quasar falls in a 1" x 30" box is ~1e-4, and any object with more than one
candidate is reported as ambiguous and given the closest-in-RA one.

Labels come from the DR16Q quasar properties catalog of Wu & Shen (2022, ApJS 263, 42),
downloadable from http://quasar.astro.illinois.edu/paper_data/DR16Q/ . Columns are
looked up by name across all FITS extensions, so the version of the file does not matter.

Output has the same layout as the sample label file:
    id,RA,DEC,Z_FIT,Z_SYS_ERR,LOGLBOL,LOGLBOL_ERR,LOGMBH,LOGMBH_ERR
RA/DEC are the catalog (SDSS) positions. Values are copied unchanged: Z_SYS_ERR = -1 and
LOGMBH/LOGLBOL = 0 mean "not measured" in the catalog; use --require-mbh to drop rows
without a usable black-hole mass.

    # IDs from the file list
    python crossmatch_labels.py --catalog DR16Q_Prop_v1.2.fits --ids qzo_list.txt --out labels.csv
    # IDs from a directory of light curves
    python crossmatch_labels.py --catalog DR16Q_Prop_v1.2.fits --ids /path/to/lightcurves --out labels.csv
    # add SDSS_NAME, match offsets and candidate counts as extra columns
    python crossmatch_labels.py ... --with-match-info
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ZONE_H = 0.0083333                 # PS1 zone height in degrees (as used by PS1, not 1/120)
LABEL_COLS = ["Z_FIT", "Z_SYS_ERR", "LOGLBOL", "LOGLBOL_ERR", "LOGMBH", "LOGMBH_ERR"]
EXTRA_COLS = ["SDSS_NAME"]


# --------------------------------------------------------------------------- IDs
def read_ids(src: Path) -> np.ndarray:
    """PS1 objIDs from a directory of <id>.csv files or a text file listing them."""
    if src.is_dir():
        names = (e.name for e in os.scandir(src) if e.is_file())
    else:
        with open(src) as fh:
            names = [ln.strip() for ln in fh]
    ids = []
    for n in names:
        n = os.path.basename(n)
        stem = n[:-4] if n.lower().endswith(".csv") else n
        if stem.isdigit():
            ids.append(int(stem))
    if not ids:
        raise SystemExit(f"no numeric IDs found in {src}")
    return np.unique(np.array(ids, dtype=np.int64))


def decode_ps1(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zone = ids // 10**13
    ra = ((ids // 10**5) % 10**8) / 1e5
    return zone.astype(np.int64), ra


# ---------------------------------------------------------------------- catalog
def read_catalog(path: Path) -> dict[str, np.ndarray]:
    from astropy.io import fits

    want = ["RA", "DEC"] + LABEL_COLS + EXTRA_COLS
    out: dict[str, np.ndarray] = {}
    nrow = None
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            names = {n.upper(): n for n in hdu.columns.names}
            for c in want:
                if c in out or c not in names:
                    continue
                col = np.asarray(hdu.data[names[c]])
                if col.ndim != 1:
                    continue
                if nrow is None:
                    nrow = len(col)
                if len(col) != nrow:
                    continue            # a different-length extension: not row-aligned
                out[c] = col.astype(str) if col.dtype.kind in "SU" else col.astype(np.float64)
    missing = [c for c in ["RA", "DEC"] + LABEL_COLS if c not in out]
    if missing:
        raise SystemExit(f"{path}: columns not found in any extension: {missing}")
    return out


# ------------------------------------------------------------------------ match
def crossmatch(zone_q, ra_q, ra_c, dec_c, ra_tol_as, dec_pad_as):
    """Return (catalog index or -1, n_candidates, dRA arcsec, dDec-outside-zone arcsec)."""
    nq = len(ra_q)
    tol = ra_tol_as / 3600.0
    pad = dec_pad_as / 3600.0

    lo_q = zone_q * ZONE_H - 90.0 - pad            # Dec bounds of each query's zone
    hi_q = (zone_q + 1) * ZONE_H - 90.0 + pad
    cosd = np.cos(np.radians(np.clip((lo_q + hi_q) / 2, -89.9, 89.9)))
    dra_max = tol / cosd                           # RA window in coordinate degrees

    order = np.argsort(ra_c, kind="stable")
    ra_s, dec_s = ra_c[order], dec_c[order]
    # pad the sorted RA array across the 0/360 seam
    wrap = max(float(dra_max.max()), 1e-6)
    left = ra_s > 360.0 - wrap
    right = ra_s < wrap
    ra_w = np.concatenate([ra_s[left] - 360.0, ra_s, ra_s[right] + 360.0])
    idx_w = np.concatenate([order[left], order, order[right]])
    dec_w = dec_c[idx_w]

    i0 = np.searchsorted(ra_w, ra_q - dra_max, side="left")
    i1 = np.searchsorted(ra_w, ra_q + dra_max, side="right")

    best = np.full(nq, -1, dtype=np.int64)
    ncand = np.zeros(nq, dtype=np.int32)
    dra_as = np.full(nq, np.nan)
    ddec_as = np.full(nq, np.nan)

    has = np.nonzero(i1 > i0)[0]
    for q in has:                                  # windows are ~0 or 1 long: cheap
        sl = slice(i0[q], i1[q])
        d = dec_w[sl]
        ok = (d >= lo_q[q]) & (d < hi_q[q])
        if not ok.any():
            continue
        dr = np.abs(ra_w[sl][ok] - ra_q[q]) * cosd[q] * 3600.0
        k = int(np.argmin(dr))
        best[q] = idx_w[sl][ok][k]
        ncand[q] = int(ok.sum())
        dra_as[q] = dr[k]
        dd = d[ok][k]
        zlo, zhi = lo_q[q] + pad, hi_q[q] - pad
        ddec_as[q] = max(zlo - dd, dd - zhi, 0.0) * 3600.0
    return best, ncand, dra_as, ddec_as


# ------------------------------------------------------------------------- main
def fmt(v: float) -> str:
    return "" if not np.isfinite(v) else repr(float(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, required=True, help="Wu & Shen DR16Q properties FITS")
    ap.add_argument("--ids", type=Path, required=True,
                    help="directory of <id>.csv light curves, or a text file listing them")
    ap.add_argument("--out", type=Path, default=Path("labels.csv"))
    ap.add_argument("--ra-tol", type=float, default=1.0, help="RA tolerance, arcsec (default 1)")
    ap.add_argument("--dec-pad", type=float, default=1.0,
                    help="allowed Dec overshoot beyond the PS1 zone, arcsec (default 1)")
    ap.add_argument("--require-mbh", action="store_true",
                    help="drop matches whose LOGMBH is missing (<=0 or NaN)")
    ap.add_argument("--with-match-info", action="store_true",
                    help="append SDSS_NAME, n_candidates, dra_arcsec, ddec_outside_zone_arcsec")
    args = ap.parse_args()

    ids = read_ids(args.ids)
    print(f"light curves:         {len(ids):,}", file=sys.stderr)
    zone, ra = decode_ps1(ids)
    bad = (zone < 0) | (zone > int(180 / ZONE_H) + 1) | (ra < 0) | (ra >= 360)
    if bad.any():
        print(f"  {bad.sum():,} IDs do not decode as PS1 objIDs; skipped", file=sys.stderr)
        ids, zone, ra = ids[~bad], zone[~bad], ra[~bad]

    cat = read_catalog(args.catalog)
    print(f"catalog rows:         {len(cat['RA']):,}", file=sys.stderr)

    best, ncand, dra, ddec = crossmatch(zone, ra, cat["RA"], cat["DEC"], args.ra_tol, args.dec_pad)
    m = best >= 0
    print(f"matched:              {m.sum():,}  ({100 * m.mean():.1f}%)", file=sys.stderr)
    print(f"  ambiguous (>1 cand): {(ncand > 1).sum():,}", file=sys.stderr)
    dup = np.bincount(best[m], minlength=len(cat["RA"]))
    print(f"  catalog rows matched by >1 light curve: {(dup > 1).sum():,}", file=sys.stderr)

    rows = np.nonzero(m)[0]
    if args.require_mbh:
        mbh = cat["LOGMBH"][best[rows]]
        keep = np.isfinite(mbh) & (mbh > 0)
        print(f"  dropped for missing LOGMBH: {(~keep).sum():,}", file=sys.stderr)
        rows = rows[keep]
    else:
        mbh = cat["LOGMBH"][best[rows]]
        nmiss = int((~(np.isfinite(mbh) & (mbh > 0))).sum())
        print(f"  with missing LOGMBH (kept; see --require-mbh): {nmiss:,}", file=sys.stderr)

    header = ["id", "RA", "DEC"] + LABEL_COLS
    if args.with_match_info:
        header += ["SDSS_NAME", "n_candidates", "dra_arcsec", "ddec_outside_zone_arcsec"]
    names = cat.get("SDSS_NAME")
    with open(args.out, "w") as fh:
        fh.write(",".join(header) + "\n")
        for q in rows:
            c = best[q]
            vals = [str(ids[q]), fmt(cat["RA"][c]), fmt(cat["DEC"][c])]
            vals += [fmt(cat[k][c]) for k in LABEL_COLS]
            if args.with_match_info:
                vals += [names[c].strip() if names is not None else "",
                         str(ncand[q]), f"{dra[q]:.3f}", f"{ddec[q]:.3f}"]
            fh.write(",".join(vals) + "\n")
    print(f"wrote {len(rows):,} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
