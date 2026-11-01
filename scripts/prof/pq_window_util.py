"""Per-window engine utilization next to the hardware throttle state, for a single-kernel device
profile (neuron-explorer parquet, e.g. from a profile of one NEFF): the busy fraction of the Tensor,
Scalar (ACT) and Vector engines in each window of W ns (default 3413, the throttle's step), and the
throttle limit k/n in force at the window's start.
Usage: python3 pq_window_util.py <parquet dir> [first window] [count] [W ns]
"""
import bisect
import os
import sys
import duckdb
def main():
    d = sys.argv[1]
    w0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 3413
    c = duckdb.connect()
    p = lambda t: f"read_parquet('{os.path.join(d, t + '.parquet')}')"  # noqa: E731
    e = c.execute(f"SELECT min(execution_index) FROM {p('Instruction')}").fetchone()[0]
    t0 = c.execute(f"SELECT min(start_ts) FROM {p('Instruction')} WHERE execution_index = {e}").fetchone()[0]
    engines = [r[0] for r in c.execute(f"SELECT DISTINCT engine FROM {p('Instruction')} WHERE execution_index = {e}").fetchall()]
    names = [x for x in ("Tensor", "Scalar", "Vector", "Pool", "GpSimd") if x in engines]
    lo, hi = t0 + w0 * W, t0 + (w0 + cnt) * W
    busy = {x: [0] * cnt for x in names}
    for eng in names:
        rows = c.execute(f"SELECT start_ts, end_ts FROM {p('Instruction')} WHERE execution_index = {e} AND engine = '{eng}' "
                         f"AND end_ts > {lo} AND start_ts < {hi} ORDER BY start_ts").fetchall()
        for a, b in rows:
            a, b = max(a, lo), min(b, hi)
            while a < b:
                i = (a - lo) // W
                e_ = min(b, lo + (i + 1) * W)
                busy[eng][i] += e_ - a
                a = e_
    thr = c.execute(f"SELECT timestamp, duration, k, n FROM {p('Throttle')} ORDER BY timestamp").fetchall()
    ts = [r[0] for r in thr]
    print(f"t0 {t0} ns, windows of {W} ns from {w0}; engines {names}")
    print("win   t_us   " + "  ".join(f"{x[:6]:>6}" for x in names) + "  throttle")
    for i in range(cnt):
        a = lo + i * W
        j = bisect.bisect_right(ts, a) - 1
        st = f"{thr[j][2]}/{thr[j][3]}" if j >= 0 and a < thr[j][0] + thr[j][1] else "-"
        print(f"{w0 + i:4d} {(a - t0) / 1e3:7.1f}  " + "  ".join(f"{busy[x][i] / W:6.2f}" for x in names) + f"  {st}")
if __name__ == "__main__":
    main()
