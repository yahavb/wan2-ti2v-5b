"""Pair the runtime inspector's device traces (.ntff) with their graphs (.neff).

The runtime (NEURON_RT_INSPECT_DEVICE_PROFILE=1) writes one per-rank dir holding
trace_info.pb plus its own NEFF dumps and one .ntff per graph execution. Its NEFF
names are NOT content hashes, so each .ntff is matched to the .neff whose filename
stem shares the longest prefix with it. The pairing table is printed so it can be
checked, and it is marked as name-based (not hash-verified).

Usage: python3 pair_systrace.py <systrace-root> <out-dir>
Writes <out-dir>/pairs.tsv  (neff, ntff_representative, exec_count, ntff_bytes)
       <out-dir>/exec_counts.json  ({neff_stem: exec_count}) for profile_all_neffs.py
       <out-dir>/rank_dir.txt      (the rank dir chosen)
"""
import json
import os
import sys


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


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

    # Rank dir = the one with the LARGEST trace_info.pb (least truncated).
    cands = []
    for d, _, files in os.walk(root):
        if "trace_info.pb" in files:
            cands.append((os.path.getsize(os.path.join(d, "trace_info.pb")), d))
    if not cands:
        print(f"NO trace_info.pb under {root} -- runtime inspector produced nothing")
        sys.exit(1)
    cands.sort(reverse=True)
    rank_dir = cands[0][1]
    print(f"rank dirs with trace_info.pb: {len(cands)}; using {rank_dir} "
          f"(trace_info.pb {cands[0][0]} bytes)")
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

    groups = {}
    for t in ntffs:
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
    print(f"paired {len(rows)} graphs (name-prefix match, NOT hash-verified); unpaired ntff: {unpaired}")
    print(f"{'neff':48} {'execs':>6} {'rep_ntff_MB':>12} {'neff_MB':>8}")
    for neff, rep, cnt, nb in rows[:40]:
        print(f"{os.path.basename(neff)[:48]:48} {cnt:>6} {nb/1e6:>12.1f} "
              f"{os.path.getsize(neff)/1e6:>8.1f}   <- {os.path.basename(rep)}")


if __name__ == "__main__":
    main()
