"""Per denoising step of the last pipeline run in a rank-0 runtime system trace: the mean device
time of the DiT block executions, to see the device's power limit tighten over a run.
Trace and parquet as for sys_phases.py (scripts/prof/rank_env.sh, then sys_to_parquet.sh).
The block NEFFs are the most frequently executed NEFFs with a mean time above min_ms; together they
must account for num_runs * steps * blocks executions.
Usage: python3 sys_block_steps.py <SystemProfileEvents.parquet> [num_runs (3)] [steps (10)]
       [blocks (30)] [min_ms (10)]
"""
import json
import sys

import duckdb


def main():
    a = sys.argv[2:] + [None] * 4
    runs, steps, blocks = int(a[0] or 3), int(a[1] or 10), int(a[2] or 30)
    min_ms = float(a[3] or 10)
    rows = duckdb.connect().execute(
        f"SELECT duration_ns, extra_attributes_json FROM read_parquet('{sys.argv[1]}') "
        f"WHERE name = 'nc_exec_running' ORDER BY start_ts").fetchall()
    ex = [(d, json.loads(j).get("model_name", "?")) for d, j in rows]
    stat = {}
    for d, n in ex:
        c, t = stat.get(n, (0, 0))
        stat[n] = (c + 1, t + d)
    want, names, total = runs * steps * blocks, set(), 0
    for n, (c, t) in sorted(stat.items(), key=lambda kv: -kv[1][0]):
        if total >= want:
            break
        if t / c > min_ms * 1e6:
            names.add(n)
            total += c
    assert total == want, f"block NEFFs ran {total} times, expected {want}"
    last = [d for d, n in ex if n in names][-steps * blocks:]
    ms = [sum(last[i * blocks:(i + 1) * blocks]) / blocks / 1e6 for i in range(steps)]
    print("block ms per step: " + " ".join(f"{x:.2f}" for x in ms)
          + f"  (step 1 -> {steps}: {100 * (ms[-1] / ms[0] - 1):+.1f}%)")


if __name__ == "__main__":
    main()
