#!/usr/bin/env python3
"""Which of the quasar light curves were already seen in pretraining?

survivor_oids.parquet lists the ZTF objects (one per field / band / CCD quadrant) that
went into the ztfsurvivors pretraining set. It has no positions, and ZTF oids are not
the Pan-STARRS1 objIDs that name the quasar files, so the two can only be joined on the
sky. This does that in three steps; each writes a file and can be rerun on its own.

  1. detect  Find which ZTF data release the oids came from. A sample of oids is looked
             up in every ztf_objects_* table on IRSA; the right release is the one whose
             ngoodobsrel and chisq equal the values stored in survivor_oids.parquet.
  2. fetch   Look up RA/Dec for all oids in that release (TAP upload + join, in chunks,
             resumable). Writes survivor_positions.parquet.
  3. match   Flag every quasar within --radius of a survivor, per band. No network.
             Writes pretrain_overlap.csv.

Positions of the quasars: sources in --labels (crossmatch_labels.py output) use their
catalog RA/Dec and a circular match. The rest use the position decoded from the PS1
objID, whose Dec is only known to its 30" zone, so they are matched on RA within
--radius and Dec inside the zone -- which admits more chance coincidences. The match
step estimates that chance rate by rerunning with the survivors shifted +-0.5 deg in
RA and reports it next to the real counts. Because quasars cluster on the sky, the
estimate runs somewhat high. Chance matches only ever flag a source as "seen" that was
not, so they cost a few evaluation objects; they never let a pretraining object through.

    pip install pyvo pandas pyarrow numpy
    python survivor_overlap.py detect --oids survivor_oids.parquet
    python survivor_overlap.py fetch  --oids survivor_oids.parquet --table <from detect>
    python survivor_overlap.py match  --ids qzo_list.txt --labels qzo_labels.csv

Run detect/fetch from a machine that can reach irsa.ipac.caltech.edu.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
ZONE_H = 0.0083333                       # PS1 zone height, degrees
BAND = {1: "g", 2: "r", 3: "i"}


# =============================================================================
# network steps
# =============================================================================
def tap():
    import pyvo
    return pyvo.dal.TAPService(TAP_URL)


def list_object_tables(svc) -> list[str]:
    q = ("SELECT table_name FROM TAP_SCHEMA.tables "
         "WHERE table_name LIKE '%ztf%' AND table_name LIKE '%objects%'")
    names = [str(n) for n in svc.run_sync(q).to_table()["table_name"]]

    def dr(n):
        digits = "".join(c if c.isdigit() else " " for c in n.split("dr")[-1]).split()
        return int(digits[0]) if digits else -1
    return sorted(names, key=dr)


def table_columns(svc, table: str) -> set[str]:
    q = f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{table}'"
    return {str(c).lower() for c in svc.run_sync(q).to_table()["column_name"]}


def lookup(svc, table: str, oids: np.ndarray, cols: list[str], sync: bool):
    """Join an uploaded list of oids against `table`; returns a DataFrame."""
    from astropy.table import Table
    up = Table({"oid": np.asarray(oids, dtype=np.int64)})
    sel = ", ".join(f"o.{c}" for c in cols)
    q = f"SELECT {sel} FROM {table} AS o JOIN TAP_UPLOAD.ids AS u ON o.oid = u.oid"
    if sync:
        res = svc.run_sync(q, uploads={"ids": up}, maxrec=len(oids) + 10)
    else:
        job = svc.submit_job(q, uploads={"ids": up}, maxrec=len(oids) + 10)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=6 * 3600)
        job.raise_if_error()
        res = job.fetch_result()
        try:
            job.delete()
        except Exception:
            pass
    return res.to_table().to_pandas()


def cmd_detect(a) -> int:
    surv = pd.read_parquet(a.oids)
    rng = np.random.default_rng(0)
    samp = surv.iloc[rng.choice(len(surv), size=min(a.sample, len(surv)), replace=False)]
    svc = tap()
    tables = a.tables or list_object_tables(svc)
    if not tables:
        print("no ztf objects tables found in TAP_SCHEMA", file=sys.stderr)
        return 1
    print(f"candidate tables: {', '.join(tables)}\n")
    print(f"{'table':28s} {'found':>7s} {'ngoodobsrel=':>13s} {'chisq=':>8s}")
    best = None
    for t in tables:
        cols = table_columns(svc, t)
        want = [c for c in ("oid", "ngoodobsrel", "chisq") if c in cols]
        if "oid" not in want:
            print(f"{t:28s}  (no oid column)")
            continue
        got = None
        # sync queries are cut off at 5 minutes; fall back to an async job (and retry)
        for attempt, sync in enumerate([not a.use_async] + [False] * a.retries):
            try:
                got = lookup(svc, t, samp.oid.values, want, sync=sync)
                break
            except Exception as e:                               # noqa: BLE001
                print(f"{t:28s}  {'sync' if sync else 'async'} query failed: {str(e)[:70]}")
                if attempt < a.retries:
                    time.sleep(10 * (attempt + 1))
        if got is None:
            continue
        m = samp.merge(got, on="oid", how="left", suffixes=("", "_t"))
        found = m["oid"].isin(got["oid"]).mean()
        n_eq = (m["ngoodobsrel"] == m.get("ngoodobsrel_t")).mean() if "ngoodobsrel" in want else np.nan
        c_eq = np.isclose(m["chisq"], m.get("chisq_t"), rtol=1e-4).mean() if "chisq" in want else np.nan
        print(f"{t:28s} {found:7.1%} {n_eq:13.1%} {c_eq:8.1%}")
        score = (n_eq if np.isfinite(n_eq) else 0) + (c_eq if np.isfinite(c_eq) else 0) + found / 10
        if best is None or score > best[0]:
            best = (score, t, found, n_eq, c_eq)
    if best:
        _, t, found, n_eq, c_eq = best
        print(f"\nbest match: {t}")
        if n_eq < 0.95:
            print("  WARNING: no release reproduces ngoodobsrel for >95% of the sample; "
                  "the oids may come from a release not on IRSA TAP, or were recomputed.")
        print(f"next: python {Path(sys.argv[0]).name} fetch --oids {a.oids} --table {t}")
    return 0


def cmd_fetch(a) -> int:
    surv = pd.read_parquet(a.oids, columns=["oid", "field", "fid"])
    oids = np.unique(surv.oid.values.astype(np.int64))
    parts = Path(a.workdir)
    parts.mkdir(parents=True, exist_ok=True)
    svc = tap()
    cols = [c for c in ("oid", "ra", "dec", "ngoodobsrel", "chisq")
            if c in table_columns(svc, a.table)]
    if not {"oid", "ra", "dec"} <= set(cols):
        print(f"{a.table} lacks oid/ra/dec columns: {cols}", file=sys.stderr)
        return 1
    chunks = [oids[i:i + a.chunk] for i in range(0, len(oids), a.chunk)]
    print(f"{len(oids):,} oids in {len(chunks)} chunks of {a.chunk:,} from {a.table}")
    t0 = time.time()
    for k, ch in enumerate(chunks):
        out = parts / f"part_{k:04d}.parquet"
        if out.exists():
            continue
        for attempt in range(1, a.retries + 1):
            try:
                df = lookup(svc, a.table, ch, cols, sync=False)
                break
            except Exception as e:                               # noqa: BLE001
                print(f"  chunk {k}: attempt {attempt} failed: {str(e)[:120]}")
                if attempt == a.retries:
                    print("  giving up; rerun the same command to resume", file=sys.stderr)
                    return 2
                time.sleep(30 * attempt)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)                                     # atomic: safe to kill
        el = time.time() - t0
        print(f"  chunk {k + 1}/{len(chunks)}: {len(df):,}/{len(ch):,} found  "
              f"({el / 60:.1f} min elapsed)")

    pos = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))],
                    ignore_index=True).drop_duplicates("oid")
    pos = surv.drop_duplicates("oid").merge(pos, on="oid", how="left")
    pos.to_parquet(a.out, index=False)
    miss = pos.ra.isna().sum()
    print(f"wrote {a.out}: {len(pos):,} oids, {miss:,} without a position")
    return 0


# =============================================================================
# match step (offline)
# =============================================================================
def read_ids(src: Path) -> np.ndarray:
    names = (e.name for e in os.scandir(src)) if src.is_dir() else (l.strip() for l in open(src))
    out = []
    for n in names:
        n = os.path.basename(n)
        s = n[:-4] if n.lower().endswith(".csv") else n
        if s.isdigit():
            out.append(int(s))
    return np.unique(np.array(out, dtype=np.int64))


def match(q_ra, q_dlo, q_dhi, q_circ, s_ra, s_dec, radius_as):
    """For each query, indices of survivors inside its search region.

    Query region: |dRA|*cos(dec) <= r and Dec in [q_dlo, q_dhi]. For circular queries
    q_dlo == q_dhi == the known Dec and the test is a true angular separation <= r.
    Returns (query_idx, survivor_idx, sep_arcsec) for every pair.
    """
    r = radius_as / 3600.0
    # bucket survivors by Dec strips of width >= the tallest query region
    strip = ZONE_H + 2 * r
    s_key = np.floor((s_dec + 90.0) / strip).astype(np.int64)
    order = np.lexsort((s_ra, s_key))
    s_key, s_ra_s, s_dec_s = s_key[order], s_ra[order], s_dec[order]
    comp = s_key * 1000.0 + s_ra_s                       # sortable (strip, RA)

    qi_all, si_all, sep_all = [], [], []
    dmid = np.clip((q_dlo + q_dhi) / 2, -89.9, 89.9)
    dra = r / np.cos(np.radians(dmid))
    k_lo = np.floor((q_dlo - r + 90.0) / strip).astype(np.int64)
    k_hi = np.floor((q_dhi + r + 90.0) / strip).astype(np.int64)
    for dk in range(0, int((k_hi - k_lo).max()) + 1):
        k = k_lo + dk
        use = k <= k_hi
        for shift in (0.0, -360.0, 360.0):               # RA seam
            lo = k * 1000.0 + (q_ra - dra + shift)
            hi = k * 1000.0 + (q_ra + dra + shift)
            i0 = np.searchsorted(comp, lo, "left")
            i1 = np.searchsorted(comp, hi, "right")
            n = np.where(use, i1 - i0, 0)
            if not n.any():
                continue
            qi = np.repeat(np.arange(len(q_ra)), n)
            si = np.concatenate([np.arange(a, b) for a, b in zip(i0[n > 0], i1[n > 0])])
            sra, sdec = s_ra_s[si] - shift, s_dec_s[si]
            dra_as = np.abs(sra - q_ra[qi]) * np.cos(np.radians(dmid[qi])) * 3600
            circ = q_circ[qi]
            ddec_c = (sdec - q_dlo[qi]) * 3600
            sep_c = np.hypot(dra_as, ddec_c)
            inside_z = (sdec >= q_dlo[qi] - r) & (sdec <= q_dhi[qi] + r)
            ok = np.where(circ, sep_c <= radius_as, (dra_as <= radius_as) & inside_z)
            qi_all.append(qi[ok]); si_all.append(order[si[ok]])
            sep_all.append(np.where(circ, sep_c, dra_as)[ok])
    if not qi_all:
        return np.array([], int), np.array([], int), np.array([])
    return np.concatenate(qi_all), np.concatenate(si_all), np.concatenate(sep_all)


def cmd_match(a) -> int:
    ids = read_ids(a.ids)
    zone = ids // 10**13
    ra = ((ids // 10**5) % 10**8) / 1e5
    dlo = zone * ZONE_H - 90.0
    dhi = (zone + 1) * ZONE_H - 90.0
    circ = np.zeros(len(ids), bool)
    if a.labels:
        lab = pd.read_csv(a.labels, dtype={"id": str}, usecols=["id", "RA", "DEC"]).dropna()
        lab = lab[lab.id.str.fullmatch(r"\d+")]
        lab_id = lab.id.astype(np.int64).values
        pos = np.searchsorted(ids, lab_id)
        pos = np.clip(pos, 0, len(ids) - 1)
        hit = ids[pos] == lab_id
        p = pos[hit]
        ra[p] = lab.RA.values[hit]
        dlo[p] = dhi[p] = lab.DEC.values[hit]
        circ[p] = True
    print(f"quasar sources: {len(ids):,}  ({circ.sum():,} with catalog positions)")

    surv = pd.read_parquet(a.positions)
    surv = surv.dropna(subset=["ra", "dec"])
    s_ra, s_dec = surv.ra.values.astype(float), surv.dec.values.astype(float)
    s_fid = surv.fid.values.astype(int)
    print(f"survivor oids with positions: {len(surv):,}")

    def run(shift_deg):
        return match(ra, dlo, dhi, circ, (s_ra + shift_deg) % 360.0, s_dec, a.radius)

    qi, si, sep = run(0.0)
    # chance-coincidence control: survivors shifted +-0.5 deg in RA, averaged
    ctrl = [run(sh)[0] for sh in (0.5, -0.5)]

    res = pd.DataFrame({"id": ids.astype(str), "precise_position": circ})
    res["n_survivor_oids"] = np.bincount(qi, minlength=len(ids))
    for f, b in BAND.items():
        sel = s_fid[si] == f
        res[f"in_pretrain_{b}"] = np.bincount(qi[sel], minlength=len(ids)) > 0
    res["in_pretrain"] = res.n_survivor_oids > 0
    best = np.full(len(ids), np.nan)
    np.fmin.at(best, qi, sep)
    res["min_sep_arcsec"] = np.round(best, 3)
    res.to_csv(a.out, index=False)

    chance = np.mean([np.bincount(c, minlength=len(ids)) > 0 for c in ctrl], axis=0)
    print(f"\nmatch radius {a.radius}\"")
    print(f"{'':32s} {'real':>10s} {'chance (est.)':>14s}")
    for name, m in (("all sources", slice(None)), ("catalog positions", circ),
                    ("ID-decoded positions", ~circ)):
        n = int(np.asarray(res.in_pretrain)[m].sum())
        c = int(round(chance[m].sum()))
        tot = int(np.ones(len(ids), bool)[m].sum())
        print(f"  {name:30s} {n:10,} {c:12,}  of {tot:,}")
    for b in ("g", "r", "i"):
        n = int(res[f"in_pretrain_{b}"].sum())
        if n:
            print(f"  in pretraining, {b} band: {n:,}")
    print(f"\nwrote {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="find the ZTF release the oids come from")
    d.add_argument("--oids", type=Path, default=Path("survivor_oids.parquet"))
    d.add_argument("--sample", type=int, default=2000)
    d.add_argument("--tables", nargs="+", default=None,
                   help="only test these tables, e.g. ztf_objects_dr21 ztf_objects_dr23")
    d.add_argument("--async", dest="use_async", action="store_true",
                   help="use async jobs from the start (sync is cut off at 5 min)")
    d.add_argument("--retries", type=int, default=2, help="async retries per table")

    f = sub.add_parser("fetch", help="look up RA/Dec for every oid")
    f.add_argument("--oids", type=Path, default=Path("survivor_oids.parquet"))
    f.add_argument("--table", required=True, help="table name printed by detect")
    f.add_argument("--chunk", type=int, default=50_000, help="oids per TAP job")
    f.add_argument("--retries", type=int, default=4)
    f.add_argument("--workdir", type=Path, default=Path("survivor_positions_parts"))
    f.add_argument("--out", type=Path, default=Path("survivor_positions.parquet"))

    m = sub.add_parser("match", help="flag quasars that were in pretraining (offline)")
    m.add_argument("--ids", type=Path, required=True, help="qzo_list.txt or a light-curve dir")
    m.add_argument("--labels", type=Path, default=None, help="qzo_labels.csv (precise positions)")
    m.add_argument("--positions", type=Path, default=Path("survivor_positions.parquet"))
    m.add_argument("--radius", type=float, default=1.5, help="arcsec (default 1.5)")
    m.add_argument("--out", type=Path, default=Path("pretrain_overlap.csv"))

    a = ap.parse_args()
    return {"detect": cmd_detect, "fetch": cmd_fetch, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
