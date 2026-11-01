"""Op-level timeline of one execution of a NEFF in a session device profile (runtime inspector
DEVICE_PROFILE_MODE=session, converted with `neuron-explorer view -d <dir> --output-format parquet`).
Takes the last execution of the NEFF that ran most often among those over 2 ms (the timed block in a
block_bench run; EXEC=<index> to choose another), and prints per HLO op (or NKI kernel) its span
from the execution start and the busy time per engine, in start order, plus the idle time of the
execution (no engine busy).
Usage: python3 pq_block_ops.py <parquet dir> [min_us (20)]   Env: EXEC
"""
import collections
import glob
import os
import sys

import duckdb


def main():
    d = sys.argv[1]
    min_us = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    inst = glob.glob(os.path.join(d, "Instruction.parquet")) or glob.glob(os.path.join(d, "Instruction*.parquet"))
    c = duckdb.connect()
    src = f"read_parquet({inst!r})"
    ex = c.execute(f"SELECT execution_index, min(start_ts), max(end_ts), count(*) FROM {src} "
                   f"GROUP BY execution_index ORDER BY execution_index").fetchall()
    print(f"{len(ex)} executions; over 2 ms (index: ms / instructions): " + " ".join(
        f"#{i}:{(b - a) / 1e6:.2f}/{n}" for i, a, b, n in ex if b - a > 2e6))
    # the NEFF executed most often among the long ones (the timed block), its last execution
    by_n = collections.Counter(n for i, a, b, n in ex if b - a > 2e6)
    n_blk = max(by_n, key=lambda n: (by_n[n], n))
    e = int(os.environ.get("EXEC", max(i for i, a, b, n in ex if n == n_blk)))
    rows = c.execute(f"SELECT start_ts, end_ts, engine, coalesce(nullif(hlo_name, ''), kernel_instruction_name, opcode) "
                     f"FROM {src} WHERE execution_index = {e} ORDER BY start_ts").fetchall()
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    ops = collections.OrderedDict()
    busy_iv = []
    for a, b, eng, name in rows:
        o = ops.setdefault(name, {"a": a, "b": b, "eng": collections.Counter()})
        o["a"], o["b"] = min(o["a"], a), max(o["b"], b)
        o["eng"][eng] += b - a
        busy_iv.append((a, b))
    busy_iv.sort()
    covered, cur_a, cur_b = 0, None, None
    for a, b in busy_iv:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    covered += cur_b - cur_a
    print(f"execution #{e}: {(t1 - t0) / 1e6:.3f} ms, some engine busy {covered / 1e6:.3f} ms, "
          f"no engine busy {(t1 - t0 - covered) / 1e6:.3f} ms; {len(rows)} instructions, {len(ops)} ops")
    print(f"{'start ms':>9} {'end ms':>8} {'span ms':>8}  busy ms per engine  op")
    for name, o in sorted(ops.items(), key=lambda kv: kv[1]["a"]):
        span = (o["b"] - o["a"]) / 1e6
        if span * 1e3 < min_us and sum(o["eng"].values()) / 1e3 < min_us:
            continue
        eng = " ".join(f"{k}={v / 1e6:.2f}" for k, v in o["eng"].most_common(4))
        print(f"{(o['a'] - t0) / 1e6:9.3f} {(o['b'] - t0) / 1e6:8.3f} {span:8.3f}  {eng:44s}  {str(name)[:90]}")


if __name__ == "__main__":
    main()
