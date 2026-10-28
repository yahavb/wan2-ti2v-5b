"""Device time of the LAST pipeline run in a rank-0 runtime system trace, split into the denoise
and VAE phases.

Trace: run inference_neuron_ti2v.py under scripts/prof/rank_env.sh with SYSTRACE_DIR=<dir>
SYSTEM_RANKS=0 PROFILE_RANKS=none, then export it with scripts/prof/sys_to_parquet.sh.
For each phase this prints wall time, device busy time (sum of nc_exec_running), idle time, the
gap distribution, the NEFFs by device time, and every idle gap > 1 ms with its offset from the
phase start and the NEFFs on either side.
Phases: the DiT block is the most frequent NEFF (blocks_per_run executions per run). The denoise
phase runs from the largest idle gap in the 2 s before the last run's first block to its last
block; the VAE phase starts at the first NEFF after that which the denoise phase never ran, and
ends with the trace.
Usage: python3 sys_phases.py <SystemProfileEvents.parquet> [num_runs (3)] [blocks_per_run (300)]
       [top_neffs (15); 0 = all]
"""
import collections
import json
import sys

import duckdb


def load(path):
    rows = duckdb.connect().execute(
        f"SELECT start_ts, end_ts, duration_ns, extra_attributes_json FROM read_parquet('{path}') "
        f"WHERE name = 'nc_exec_running' ORDER BY start_ts").fetchall()
    return [(a, b, d, json.loads(j).get("model_name", "?")) for a, b, d, j in rows]


def split(ex, runs, per_run):
    """Indices of the last run's first denoise execution and first VAE execution."""
    block, n = collections.Counter(x[3] for x in ex).most_common(1)[0]
    print(f"{len(ex)} executions; most frequent NEFF {block} x{n} (expect {runs * per_run})")
    idx = [i for i, x in enumerate(ex) if x[3] == block]
    first, last = idx[(runs - 1) * per_run], idx[runs * per_run - 1]
    lo = first
    while lo > 0 and ex[first][0] - ex[lo - 1][1] < 2e9:
        lo -= 1
    gaps = [(ex[i][0] - ex[i - 1][1], i) for i in range(lo + 1, first + 1)]
    start = max(gaps)[1] if gaps else first
    dit = {x[3] for x in ex[start:last + 1]}
    vae = next(i for i in range(last + 1, len(ex)) if ex[i][3] not in dit)
    return start, vae


def report(name, seg, top):
    t0 = seg[0][0]
    wall = (seg[-1][1] - t0) / 1e9
    busy = sum(x[2] for x in seg) / 1e9
    gaps = [max(0, y[0] - x[1]) for x, y in zip(seg, seg[1:])]
    srt = sorted(gaps)
    q = lambda f: srt[int(f * (len(srt) - 1))] / 1e3 if srt else 0.0
    big = [i for i, g in enumerate(gaps) if g > 1e6]
    big_sum = sum(gaps[i] for i in big)
    print(f"\n== {name}: wall {wall:.3f} s, device busy {busy:.3f} s ({100 * busy / wall:.1f}%), "
          f"idle {wall - busy:.3f} s, {len(seg)} executions")
    print(f"   gaps us p50 {q(.5):.1f} p90 {q(.9):.1f} p99 {q(.99):.1f}; {len(big)} gaps > 1 ms sum "
          f"{big_sum / 1e9:.3f} s; smaller gaps sum {(sum(gaps) - big_sum) / 1e9:.3f} s")
    by = collections.defaultdict(lambda: [0, 0])
    for x in seg:
        by[x[3]][0] += x[2]
        by[x[3]][1] += 1
    order = sorted(by.items(), key=lambda kv: -kv[1][0])
    shown = order if top == 0 else order[:top]
    for nm, (t, c) in shown:
        print(f"  {t / 1e6:9.1f} ms {100 * t / 1e9 / wall:5.1f}% of wall  n={c:5d}  avg {t / c / 1e3:8.1f} us  {nm}")
    rest = order[len(shown):]
    if rest:
        print(f"  {sum(v[0] for _, v in rest) / 1e6:9.1f} ms in the other {len(rest)} NEFFs")
    print("  idle gaps > 1 ms (offset from the phase start):")
    for i in big:
        x, y = seg[i], seg[i + 1]
        print(f"    +{(x[1] - t0) / 1e6:8.1f} ms: {gaps[i] / 1e6:6.1f} ms after exec #{i} ...{x[3][-12:]} "
              f"({x[2] / 1e3:.0f} us), before ...{y[3][-12:]}")


def main():
    path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    per_run = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    ex = load(path)
    start, vae = split(ex, runs, per_run)
    print(f"denoise -> VAE handoff gap {(ex[vae][0] - ex[vae - 1][1]) / 1e6:.1f} ms")
    report("denoise (last run)", ex[start:vae], top)
    report("VAE (last run)", ex[vae:], top)


if __name__ == "__main__":
    main()
