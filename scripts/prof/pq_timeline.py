#!/usr/bin/env python3
"""Emit a TIME SERIES of execution from an ingested trace, not aggregates.

Aggregates say WHERE time went; they cannot say in what ORDER, nor where the engines waited.
This prints the sequence: instructions in timestamp order, occupancy per time bucket, and the
largest gaps -- which is what identifies whether a graph is bottlenecked on one long operation or
on many small ones with dead time between them.

IT DISCOVERS THE SCHEMA AND NEVER ASSUMES COLUMN NAMES. Table and column names differ across
profiler versions, so guessing them produces either a crash or -- worse -- a confident query
against the wrong column. Every column used here is matched from the actual parquet header at
runtime and PRINTED, so the report says exactly which fields it read. If a needed field is
absent it says so and stops rather than substituting something plausible.

Usage: pq_timeline.py <parquet-dir> [label] [--head N] [--buckets N]
"""
import glob
import os
import sys

import duckdb

TIME_HINTS = ("start", "timestamp", "begin", "ts", "time_ns", "cycle")
DUR_HINTS = ("duration", "elapsed", "active_time", "busy", "latency", "cost")
ENG_HINTS = ("engine", "queue", "unit", "core", "nc_idx")
NAME_HINTS = ("hlo_name", "opcode", "name", "instruction", "op")
SRC_HINTS = ("source_location", "nki_source", "debug_info")


def pick(cols, hints, exclude=()):
    """First column whose name contains a hint, in hint order. Returns None if none match."""
    low = {c.lower(): c for c in cols}
    for h in hints:
        for lc, orig in low.items():
            if h in lc and not any(x in lc for x in exclude):
                return orig
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pqdir = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    head_n = 40
    buckets = 60
    for i, a in enumerate(sys.argv):
        if a == "--head" and i + 1 < len(sys.argv):
            head_n = int(sys.argv[i + 1])
        if a == "--buckets" and i + 1 < len(sys.argv):
            buckets = int(sys.argv[i + 1])

    con = duckdb.connect()
    files = sorted(glob.glob(os.path.join(pqdir, "*.parquet")))
    if not files:
        print("no parquet in %s -- the ingest produced nothing, so there is no series to print" % pqdir)
        return 1

    print("=" * 100)
    print("TIMELINE  %s" % (label or pqdir))
    print("=" * 100)
    print("tables found (name, rows, columns):")
    tables = {}
    for f in files:
        t = os.path.splitext(os.path.basename(f))[0]
        try:
            cols = [d[0] for d in con.execute(
                "SELECT * FROM read_parquet('%s') LIMIT 0" % f).description]
            n = con.execute("SELECT count(*) FROM read_parquet('%s')" % f).fetchone()[0]
        except Exception as e:                                 # noqa: BLE001
            print("  %-28s UNREADABLE (%s)" % (t, type(e).__name__))
            continue
        tables[t] = (f, cols, n)
        print("  %-28s %12s rows   %s" % (t, "{:,}".format(n), ", ".join(cols[:8])
                                          + (" ..." if len(cols) > 8 else "")))

    # The instruction-level table is the only one that can carry a per-op timeline.
    inst = None
    for cand in ("Instruction", "Instructions", "instruction"):
        if cand in tables:
            inst = cand
            break
    if inst is None:
        inst = max(tables, key=lambda t: tables[t][2]) if tables else None
        print("\nno table literally named Instruction; using the largest table: %s" % inst)
    if inst is None:
        return 1

    f, cols, nrows = tables[inst]
    tcol = pick(cols, TIME_HINTS, exclude=("duration", "elapsed", "wait", "total"))
    dcol = pick(cols, DUR_HINTS)
    ecol = pick(cols, ENG_HINTS)
    ncol = pick(cols, NAME_HINTS)
    scol = pick(cols, SRC_HINTS)
    print("\ncolumns SELECTED from %s (matched against the real header, not assumed):" % inst)
    for k, v in (("time", tcol), ("duration", dcol), ("engine", ecol),
                 ("name", ncol), ("source", scol)):
        print("  %-9s -> %s" % (k, v if v else "NOT PRESENT"))
    if tcol is None:
        print("\nNO TIME COLUMN in %s. A time series is not derivable from this table; the "
              "available columns are printed above. Stopping rather than inventing an ordering."
              % inst)
        return 1

    sel = [tcol] + [c for c in (dcol, ecol, ncol, scol) if c]
    q = "SELECT %s FROM read_parquet('%s') WHERE %s IS NOT NULL ORDER BY %s" % (
        ", ".join('"%s"' % c for c in sel), f, tcol, tcol)

    print("\n--- span ---")
    lo, hi = con.execute('SELECT min("%s"), max("%s") FROM read_parquet(\'%s\')'
                         % (tcol, tcol, f)).fetchone()
    print("  %s from %s to %s  (span %s)" % (tcol, lo, hi, (hi - lo) if hi is not None else "?"))

    print("\n--- first %d instructions in TIME ORDER ---" % head_n)
    rows = con.execute(q + " LIMIT %d" % head_n).fetchall()
    print("  " + "  ".join("%-22s" % c for c in sel))
    for r in rows:
        print("  " + "  ".join("%-22s" % (str(x)[:22]) for x in r))

    if dcol:
        print("\n--- longest %d single instructions (the ones to look at first) ---" % 15)
        for r in con.execute(
                'SELECT %s FROM read_parquet(\'%s\') ORDER BY "%s" DESC LIMIT 15'
                % (", ".join('"%s"' % c for c in sel), f, dcol)).fetchall():
            print("  " + "  ".join("%-22s" % (str(x)[:22]) for x in r))

        # Occupancy per bucket: busy time vs bucket width shows dead time directly.
        # SUMMED busy, not occupancy: engines run concurrently, so the total can exceed the
        # bucket width. Same caveat as per-engine percentages summing past 100% -- these are
        # overlapping busy times, not a partition of wall time. What the column IS good for is
        # the SHAPE: a bucket far below its neighbours is dead time.
        print("\n--- summed busy per bucket over %d equal time buckets ---" % buckets)
        print("  (engines overlap, so summed busy CAN exceed the bucket width; read the shape,")
        print("   not the absolute -- a low bucket is where the machine waited)")
        print("  bucket        instrs   summed_busy   bucket_width   ratio")
        width = (hi - lo) / buckets if hi and hi > lo else 0
        if width:
            bq = ('SELECT floor(("%s" - %s) / %s) AS b, count(*), sum("%s") '
                  'FROM read_parquet(\'%s\') GROUP BY b ORDER BY b' % (tcol, lo, width, dcol, f))
            for b, cnt, busy in con.execute(bq).fetchall():
                if b is None:
                    continue
                pct = (100.0 * busy / width) if busy is not None and width else 0.0
                print("  %6s %12s %11s %15.1f %7.1f" % (int(b), "{:,}".format(cnt),
                                                        round(busy or 0, 1), width, pct))

    if ecol and dcol:
        print("\n--- busy time by %s (shares are the stable signal, absolutes are not) ---" % ecol)
        for e, cnt, busy in con.execute(
                'SELECT "%s", count(*), sum("%s") FROM read_parquet(\'%s\') '
                'GROUP BY 1 ORDER BY 3 DESC LIMIT 20' % (ecol, dcol, f)).fetchall():
            print("  %-30s %12s instrs   busy %s" % (str(e)[:30], "{:,}".format(cnt),
                                                     round(busy or 0, 1)))

    if ncol and dcol:
        print("\n--- busy time by %s, top 20 ---" % ncol)
        for nm, cnt, busy in con.execute(
                'SELECT "%s", count(*), sum("%s") FROM read_parquet(\'%s\') '
                'GROUP BY 1 ORDER BY 3 DESC LIMIT 20' % (ncol, dcol, f)).fetchall():
            print("  %-52s %10s   busy %s" % (str(nm)[:52], "{:,}".format(cnt),
                                              round(busy or 0, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
