"""Pair the runtime inspector's device traces (.ntff) with their graphs (.neff).

The runtime (NEURON_RT_INSPECT_DEVICE_PROFILE=1) writes one per-rank dir holding
trace_info.pb plus its own NEFF dumps and .ntff device traces. The two share a numeric
id: neff_<id>.neff <-> <id>_instid_<i>_vnc_<v>.ntff (Neuron Explorer docs). Anything
without that id falls back to the longest shared filename prefix. The pairing table is
printed so it can be checked.

Usage: python3 pair_systrace.py <systrace-root> <out-dir>
Writes <out-dir>/pairs.tsv  (neff, ntff_representative, exec_count, ntff_bytes)
       <out-dir>/exec_counts.json  ({neff_stem: exec_count}) for profile_all_neffs.py
       <out-dir>/rank_dir.txt      (the rank dir chosen)
"""
import json
import os
import re
import sys


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def neff_id(p):
    m = re.match(r"(?:neff_)?(\d+)", stem(p))
    return m.group(1) if m else None


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    # Rank dir = the one holding device traces (most .ntff bytes), then the LARGEST
    # trace_info.pb (least truncated). Only PROFILE_RANKS write .ntff.
    cands = []
    for d, _, files in os.walk(root):
        if "trace_info.pb" in files:
            nt = sum(os.path.getsize(os.path.join(d2, f)) for d2, _, fs in os.walk(d)
                     for f in fs if f.endswith(".ntff"))
            cands.append((nt, os.path.getsize(os.path.join(d, "trace_info.pb")), d))
    if not cands:
        print(f"NO trace_info.pb under {root} -- runtime inspector produced nothing")
        sys.exit(1)
    cands.sort(reverse=True)
    rank_dir = cands[0][2]
    print(f"rank dirs with trace_info.pb: {len(cands)}; using {rank_dir} "
          f"(ntff {cands[0][0]/1e6:.0f} MB, trace_info.pb {cands[0][1]} bytes)")
    open(os.path.join(out, "rank_dir.txt"), "w").write(rank_dir + "\n")

    neffs, ntffs = [], []
    for d, _, files in os.walk(rank_dir):
        for f in files:
            p = os.path.join(d, f)
            if f.endswith(".neff"):
                neffs.append(p)
            elif f.endswith(".ntff"):
                ntffs.append(p)
    print(f"device files: {len(neffs)} .neff, {len(ntffs)} .ntff")
    if not neffs or not ntffs:
        print("no device-level pairs (DEVICE_PROFILE may be unsupported for this workload)")
        open(os.path.join(out, "pairs.tsv"), "w").close()
        json.dump({}, open(os.path.join(out, "exec_counts.json"), "w"))
        return

    by_id = {neff_id(n): n for n in neffs if neff_id(n)}
    groups = {}
    for t in ntffs:
        if neff_id(t) in by_id:
            groups.setdefault(by_id[neff_id(t)], []).append(t)
            continue
        ts = stem(t)
        best = max(neffs, key=lambda n: (common_prefix(stem(n), ts), -len(stem(n))))
        if common_prefix(stem(best), ts) == 0:
            continue
        groups.setdefault(best, []).append(t)

    rows = []
    for neff, ts in groups.items():
        rep = max(ts, key=os.path.getsize)
        rows.append((neff, rep, len(ts), os.path.getsize(rep)))
    rows.sort(key=lambda r: -r[3])

    with open(os.path.join(out, "pairs.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    json.dump({stem(r[0]): r[2] for r in rows}, open(os.path.join(out, "exec_counts.json"), "w"))

    unpaired = len(ntffs) - sum(r[2] for r in rows)
    print(f"paired {len(rows)} graphs (numeric-id match, else name prefix); unpaired ntff: {unpaired}")
    print(f"{'neff':48} {'execs':>6} {'rep_ntff_MB':>12} {'neff_MB':>8}")
    for neff, rep, cnt, nb in rows[:40]:
        print(f"{os.path.basename(neff)[:48]:48} {cnt:>6} {nb/1e6:>12.1f} "
              f"{os.path.getsize(neff)/1e6:>8.1f}   <- {os.path.basename(rep)}")


if __name__ == "__main__":
    main()
