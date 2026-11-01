"""Coarse timeline of one execution in a session device profile (see pq_block_ops.py): per time
bucket, the active fraction of each engine (from ActiveTime), the PE utilization limit in force
(from Throttle; 1.0 = none) and the op the Tensor engine spent most time on.
Usage: python3 pq_timeline_buckets.py <parquet dir> <execution index> [bucket_us (200)]
"""
import collections
import os
import sys

import duckdb


def main():
    d, e = sys.argv[1], int(sys.argv[2])
    bk = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 200e3
    c = duckdb.connect()
    p = lambda t: f"read_parquet('{os.path.join(d, t + '.parquet')}')"  # noqa: E731
    a, b = c.execute(f"SELECT min(start_ts), max(end_ts) FROM {p('Instruction')} WHERE execution_index = {e}").fetchone()
    nb = int((b - a) // bk) + 1
    act = collections.defaultdict(lambda: [0.0] * nb)
    for s, t, eng in c.execute(f"SELECT start_ts, end_ts, engine FROM {p('ActiveTime')} "
                               f"WHERE end_ts > {a} AND start_ts < {b}").fetchall():
        s, t = max(s, a), min(t, b)
        while s < t:
            i = int((s - a) // bk)
            edge = a + (i + 1) * bk
            act[eng][i] += min(t, edge) - s
            s = min(t, edge)
    lim = [1.0] * nb
    for ts, dur, k, n in c.execute(f"SELECT timestamp, duration, k, n FROM {p('Throttle')} "
                                   f"WHERE timestamp >= {a} AND timestamp < {b}").fetchall():
        s, t = ts, min(ts + dur, b)
        while s < t:
            i = int((s - a) // bk)
            edge = a + (i + 1) * bk
            lim[i] -= (1 - k / n) * (min(t, edge) - s) / bk  # time-weighted limit
            s = min(t, edge)
    top = [collections.Counter() for _ in range(nb)]
    for s, t, name in c.execute(
            f"SELECT start_ts, end_ts, coalesce(nullif(hlo_name, ''), nullif(nki_source_location, ''), opcode) "
            f"FROM {p('Instruction')} WHERE execution_index = {e} AND engine = 'Tensor'").fetchall():
        top[int((s - a) // bk)][str(name).split("/")[-1][:60]] += t - s
    engs = ["tensor", "scalar", "vector", "gpsimd", "sync"]
    print(f"{'t ms':>6} " + " ".join(f"{x[:6]:>6}" for x in engs) + "  PE-lim  top Tensor op")
    for i in range(nb):
        op = top[i].most_common(1)[0][0] if top[i] else ""
        print(f"{i * bk / 1e6:6.2f} " + " ".join(f"{act[x][i] / bk:6.2f}" for x in engs) + f"  {lim[i]:5.2f}   {op}")


if __name__ == "__main__":
    main()
