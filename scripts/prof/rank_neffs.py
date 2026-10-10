"""Rank EVERY captured NEFF by FIXABILITY (time-weighted idle), not wall time, and
apply the ENGINE-MIX GATE to class each one. This is Step A+B of the "Hotspot hunt"
procedure in SKILL.md. Input = the dir of per-NEFF summary-json (<hash>.json) that
step-5 produced. Emits neff_ranking.txt content to stdout.

A NEFF is a fixable copy ONLY if matmul≈0 AND tensor%<~15 AND dma dominates; a NEFF
with matmul>0 AND tensor>~40% is REAL COMPUTE (skip it even if wall-time-huge). Rank
the copy/SWDGE/transpose classes by waste_us to find the fixable hotspot.

Usage: python3 rank_neffs.py <dir-of-summary-json>
Verified on RF fs1440 (trn3/SDK2.31): surfaced vae.py SiLU/norm copies as the
fixable set while correctly leaving the 59-75%-tensor DiT GEMMs off the fix list.
"""
import sys
import os
import json
import glob


def dom(d):
    ns = [v for v in d.values() if isinstance(v, dict)] if isinstance(d, dict) else d
    return max(ns or [d], key=lambda x: (x.get("total_time", 0) or 0))


def classify(r):
    # ENGINE-MIX GATE — the source line does NOT decide this; the engine mix does.
    if r["matmul"] < 500 and r["tensor"] < 15 and (r["sw_dyn"] >= 30 or r["static_dma"] > 5000):
        return "COPY-FIXABLE"
    if r["sw_dyn"] >= 30 and r["matmul"] >= 500:
        return "SWDGE-heavy"        # gather/scatter riding inside a matmul NEFF
    if r["tflops"] > 1e8 and r["tensor"] < 50:
        return "TRANSPOSE-heavy"
    if r["matmul"] > 0 and r["tensor"] >= 40:
        return "COMPUTE (not fixable)"
    return "MIXED"


def main():
    jdir = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(jdir, "*.json"))):
        try:
            n = dom(json.load(open(f)))
        except Exception:
            continue
        g = lambda k: float(n.get(k, 0) or 0)
        pct = lambda k: g(k + "_percent") * 100.0
        r = dict(
            h=os.path.basename(f)[:-5],
            us=g("total_time") * 1e6,
            tensor=pct("tensor_engine_active_time"),
            vector=pct("vector_engine_active_time"),
            scalar=pct("scalar_engine_active_time"),
            gpsimd=pct("gpsimd_engine_active_time"),
            dma=pct("dma_active_time"),
            sw_dyn=pct("software_dynamic_dma_packet"),
            matmul=g("matmul_instruction_count"),
            static_dma=g("static_dma_packet_count"),
            tflops=g("transpose_flops"),
        )
        r["maxeng"] = max(r["tensor"], r["vector"], r["scalar"], r["gpsimd"])
        r["waste_us"] = r["us"] * max(0.0, 100 - r["maxeng"]) / 100.0
        r["class"] = classify(r)
        rows.append(r)
    if not rows:
        print("NO summary-json found — capture failed."); return

    tot = sum(x["us"] for x in rows) or 1.0
    rows.sort(key=lambda x: -x["waste_us"])

    print(f"=== NEFF FIXABILITY RANKING ({len(rows)} NEFFs, {tot/1000:.1f}ms capture sum) ===")
    print("Ranked by waste_us = us * idle_frac (time no engine is busy). Fix the top")
    print("COPY-FIXABLE / SWDGE-heavy / TRANSPOSE-heavy entry — NOT rank-1 if it is COMPUTE.")
    print(f"{'neff':16}{'us':>8}{'waste':>8}{'maxeng':>7}{'tensor':>7}{'dma':>6}{'swdyn':>6}"
          f"{'mm':>7}{'statDMA':>9}  class")
    for r in rows[:30]:
        print(f"{r['h'][:14]:16}{r['us']:>8.0f}{r['waste_us']:>8.0f}{r['maxeng']:>7.0f}"
              f"{r['tensor']:>7.0f}{r['dma']:>6.0f}{r['sw_dyn']:>6.0f}{r['matmul']:>7.0f}"
              f"{r['static_dma']:>9.0f}  {r['class']}")

    print("\n=== TOP FIXABLE (COPY / SWDGE / TRANSPOSE only — these are the pairs to parquet) ===")
    fix = [r for r in rows if r["class"] in ("COPY-FIXABLE", "SWDGE-heavy", "TRANSPOSE-heavy")]
    for i, r in enumerate(fix[:12], 1):
        print(f"  F{i:<2} {r['h']}  {r['waste_us']:.0f}us waste  {r['class']}")
    if not fix:
        print("  (none — no copy/gather/transpose NEFF; bottleneck is real compute or between-NEFF gaps)")


if __name__ == "__main__":
    main()
