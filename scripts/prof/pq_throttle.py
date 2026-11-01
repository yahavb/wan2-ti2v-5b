"""Hardware throttle events of one execution in a session device profile (see pq_block_ops.py),
attributed to the op the Tensor engine was running when each event began, plus the Tensor engine's
idle gaps. Usage: python3 pq_throttle.py <parquet dir> <execution index>
"""
import bisect
import collections
import os
import sys

import duckdb


def main():
    d, e = sys.argv[1], int(sys.argv[2])
    c = duckdb.connect()
    p = lambda t: f"read_parquet('{os.path.join(d, t + '.parquet')}')"  # noqa: E731
    a, b = c.execute(f"SELECT min(start_ts), max(end_ts) FROM {p('Instruction')} WHERE execution_index = {e}").fetchone()
    pe = c.execute(f"SELECT start_ts, end_ts, coalesce(nullif(hlo_name, ''), nullif(nki_source_location, ''), opcode) "
                   f"FROM {p('Instruction')} WHERE execution_index = {e} AND engine = 'Tensor' ORDER BY start_ts").fetchall()
    starts = [r[0] for r in pe]
    thr = c.execute(f"SELECT timestamp, duration, type, k, n FROM {p('Throttle')} "
                    f"WHERE timestamp >= {a} AND timestamp < {b} ORDER BY timestamp").fetchall()
    print(f"execution {e}: {(b - a) / 1e6:.3f} ms, {len(pe)} Tensor instructions, {len(thr)} throttle events")
    agg = collections.defaultdict(lambda: [0, 0])
    for ts, dur, typ, k, n in thr:
        i = max(0, bisect.bisect_right(starts, ts) - 1)
        name = str(pe[i][2])[:70] if pe else "?"
        key = (f"{typ} {k}/{n}", name)
        agg[key][0] += 1
        agg[key][1] += dur
    for (typ, name), (cnt, dur) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"  {typ:14s} x{cnt:4d} {dur / 1e6:7.3f} ms  {name}")
    act = c.execute(f"SELECT start_ts, end_ts FROM {p('ActiveTime')} WHERE engine = 'tensor' "
                    f"AND end_ts > {a} AND start_ts < {b} ORDER BY start_ts").fetchall()
    gaps = [y[0] - x[1] for x, y in zip(act, act[1:]) if y[0] > x[1]]
    for lo, hi in ((0, 1e3), (1e3, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e12)):
        g = [x for x in gaps if lo <= x < hi]
        print(f"  Tensor idle gaps {lo / 1e3:6.0f}-{hi / 1e3:<6.0f} us: n={len(g):5d} sum {sum(g) / 1e6:.3f} ms")


if __name__ == "__main__":
    main()
