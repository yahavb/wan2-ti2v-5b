#!/usr/bin/env python3
"""Turn a decoded NEFF+NTFF pair into a trace you can open in ui.perfetto.dev.

WHY THIS EXISTS. The runtime's system trace converts to Perfetto directly, but it shows WHEN
graphs ran, not what ran INSIDE one. What ran inside a graph is in the per-NEFF pair, and the
profiler may not offer a perfetto output for a pair -- only summary and table formats are
confirmed. So this builds the trace from the instruction table itself, which is deterministic
because we control the conversion: one slice per instruction, one track per engine.

Perfetto reads the Chrome Trace JSON format, so that is what this writes (gzipped -- the viewer
takes .gz directly). Each instruction becomes an X event with a start and a duration; each engine
becomes a named track, so the picture is per-engine occupancy over time with the ops in place.

UNITS ARE STATED, NOT GUESSED. Chrome Trace timestamps are MICROSECONDS. The table's units are
not knowable from the schema, so --scale converts and the script PRINTS what it assumed along
with the resulting span, which is the number to sanity-check: if the span is wildly wrong the
scale is wrong. Default 1e-3 treats the source as nanoseconds.

Usage: pq_to_perfetto.py <parquet-dir> <out.json.gz> [--scale 0.001] [--limit N]
"""
import glob
import gzip
import json
import os
import sys

import duckdb

TIME_HINTS = ("start", "timestamp", "begin", "ts", "time_ns", "cycle")
DUR_HINTS = ("duration", "elapsed", "active_time", "busy", "latency", "cost")
ENG_HINTS = ("engine", "queue", "unit", "core", "nc_idx")
NAME_HINTS = ("hlo_name", "opcode", "name", "instruction", "op")
SRC_HINTS = ("source_location", "nki_source", "debug_info")


def pick(cols, hints, exclude=()):
    low = {c.lower(): c for c in cols}
    for h in hints:
        for lc, orig in low.items():
            if h in lc and not any(x in lc for x in exclude):
                return orig
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pqdir, out = sys.argv[1], sys.argv[2]
    scale = 1e-3
    limit = 0
    for i, a in enumerate(sys.argv):
        if a == "--scale" and i + 1 < len(sys.argv):
            scale = float(sys.argv[i + 1])
        if a == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    con = duckdb.connect()
    files = sorted(glob.glob(os.path.join(pqdir, "*.parquet")))
    if not files:
        print("no parquet in %s -- nothing to convert" % pqdir)
        return 1
    tables = {}
    for f in files:
        t = os.path.splitext(os.path.basename(f))[0]
        try:
            cols = [d[0] for d in con.execute(
                "SELECT * FROM read_parquet('%s') LIMIT 0" % f).description]
            tables[t] = (f, cols)
        except Exception:                                          # noqa: BLE001
            continue

    inst = next((c for c in ("Instruction", "Instructions", "instruction") if c in tables), None)
    if inst is None:
        print("no instruction table among: %s" % ", ".join(sorted(tables)))
        return 1
    f, cols = tables[inst]
    tcol = pick(cols, TIME_HINTS, exclude=("duration", "elapsed", "wait", "total"))
    dcol = pick(cols, DUR_HINTS)
    ecol = pick(cols, ENG_HINTS)
    ncol = pick(cols, NAME_HINTS)
    scol = pick(cols, SRC_HINTS)
    print("columns used (matched from the real header):")
    for k, v in (("time", tcol), ("duration", dcol), ("engine", ecol),
                 ("name", ncol), ("source", scol)):
        print("  %-9s -> %s" % (k, v or "NOT PRESENT"))
    if tcol is None or dcol is None:
        print("\nNeed BOTH a time and a duration column to place slices. Missing one, so a "
              "timeline cannot be built from this table -- stopping rather than emitting "
              "something that looks like a trace but is not.")
        return 1

    sel = [tcol, dcol] + [c for c in (ecol, ncol, scol) if c]
    q = "SELECT %s FROM read_parquet('%s') WHERE %s IS NOT NULL ORDER BY %s" % (
        ", ".join('"%s"' % c for c in sel), f, tcol, tcol)
    if limit:
        q += " LIMIT %d" % limit
    rows = con.execute(q).fetchall()
    if not rows:
        print("instruction table has no rows with a timestamp")
        return 1

    # One track per engine so the result reads as per-engine occupancy. Track ids are assigned in
    # first-seen order, which is time order, so the tracks appear roughly in execution order.
    tids = {}
    events = []
    for r in rows:
        ts = float(r[0]) * scale
        dur = float(r[1] or 0) * scale
        eng = str(r[2]) if ecol else "engine"
        name = str(r[3]) if ncol else "instr"
        src = str(r[4]) if scol else None
        if eng not in tids:
            tids[eng] = len(tids) + 1
        ev = {"name": name[:120], "ph": "X", "ts": ts, "dur": dur, "pid": 1, "tid": tids[eng]}
        if src:
            # Perfetto shows args when a slice is selected: this is what ties a slice to source.
            ev["args"] = {"source": src[:200]}
        events.append(ev)

    for eng, tid in tids.items():
        events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
                       "args": {"name": eng[:60]}})
    events.append({"name": "process_name", "ph": "M", "pid": 1, "tid": 0,
                   "args": {"name": "NEFF %s" % os.path.basename(pqdir)}})

    doc = {"traceEvents": events, "displayTimeUnit": "ns"}
    with gzip.open(out, "wt") as fh:
        json.dump(doc, fh)

    span_us = max(e["ts"] + e["dur"] for e in events if e["ph"] == "X") - \
        min(e["ts"] for e in events if e["ph"] == "X")
    print("\nwrote %s  (%.1f MB gz)" % (out, os.path.getsize(out) / 1e6))
    print("  slices        %s" % "{:,}".format(len(rows)))
    print("  tracks        %d  (%s)" % (len(tids), ", ".join(list(tids)[:8])))
    print("  scale applied %g  -> span %.3f ms" % (scale, span_us / 1000.0))
    print("  SANITY-CHECK THE SPAN: if it is not roughly the graph's known duration, --scale is")
    print("  wrong. The table's time unit is not knowable from the schema, so it is an argument.")
    print("  Open it at https://ui.perfetto.dev (it reads .gz directly).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
