#!/usr/bin/env bash
# 3-run Neuron benchmark for Wan2.2-TI2V-5B (run from the repo root inside the job pod).
#
#   [1+2] CLEAN   : torchrun, NUM_RUNS=$CLEAN_RUNS, NO profiler. Run 1 compiles/loads (not
#                   timed); runs 2..N are the real latency.
#   [3a]  PROFILED: torchrun, runtime inspector (system + device profile) on PROFILE_RANKS
#                   only, NUM_STEPS=$PROFILE_STEPS. Gives trace_info.pb + per-graph .ntff and
#                   the profiler-overhead delta.
#   [3b]  system timeline -> perfetto
#   [4]   per-graph summary-json for every paired .neff/.ntff -> engine table + fixability
#         ranking; summary-text on the dominant graph; preserve top pairs in pairs/rankNN_*
#   [4b]  --single-io sweep over the NEFF cache (capped; expect 0/N on collective graphs)
#   [5]   parquet ingest + duckdb op/DMA report on the preserved pairs (analyze_trace.sh)
#
# Everything is written to local disk under $RESULTS and tarred to $RESULTS.tar.gz (no S3).
# No `set -e`: a failed profiling stage must not cost the clean latency or the video.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
TP="${TP_DEGREE:-4}"
CLEAN_RUNS="${CLEAN_RUNS:-3}"
PROFILE_STEPS="${PROFILE_STEPS:-2}"
MAX_SUMMARY="${MAX_SUMMARY:-40}"
MAX_PAIRS="${MAX_PAIRS:-3}"
SWEEP_MAX="${SWEEP_MAX:-6}"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS="/tmp/results/wan2_tp${TP}_${TS}"
mkdir -p "$RESULTS"
echo "$RESULTS" > /tmp/results/LATEST
LAUNCH=(torchrun --nproc_per_node="$TP" --master_port=29500)

PROF=neuron-explorer; command -v "$PROF" >/dev/null || PROF=neuron-profile
echo "profiler CLI: $PROF ($("$PROF" --version 2>&1 | head -1))"

# ── PROVENANCE ──
GIT_SHA=$(git rev-parse HEAD); GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
{
  echo "┌── PROVENANCE ──"
  echo "│ branch : $GIT_BRANCH"
  echo "│ commit : $GIT_SHA"
  echo "│ subject: $(git log -1 --pretty=%s)"
  echo "│ TP=$TP DIT_FAST=${DIT_FAST:-} FAST_ATTN_IMPL=${FAST_ATTN_IMPL:-} LNC=${LNC_CONFIG:-}"
  echo "│ feature: $(python3 -c 'import sys; sys.path.insert(0,"."); import models.fast_dit as f; print("fast_dit attn="+f.FAST_ATTN_IMPL, "fullgraph="+str(f.FAST_FULLGRAPH))' 2>&1 | tail -1)"
  echo "│ results: $RESULTS"
  echo "└────────────────"
} | tee "$RESULTS/provenance.txt"

# s/step of every "Run i/N denoising: X s (Y s/step)" line in a log
steps_of() { grep -oE "denoising: [0-9.]+s \([0-9.]+s/step\)" "$1" | grep -oE "[0-9.]+s/step" | tr -d 's/step'; }

archive() {
  echo "=== archiving $RESULTS ==="
  tar -czf "$RESULTS.tar.gz" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")" 2>/dev/null
  echo "ARCHIVE: $RESULTS.tar.gz ($(wc -c < "$RESULTS.tar.gz" 2>/dev/null) bytes)"
}
trap archive EXIT

# ═══ [0] const-max kernel parity (one core) when that kernel is selected ═══
if [ "${FAST_ATTN_IMPL:-}" = "constmax" ]; then
  echo "=== [0] const-max parity vs exact attention (one core) ==="
  NEURON_RT_VISIBLE_CORES=0 timeout 1800 python3 scripts/prof/const_max_parity.py 2>&1 \
    | grep -vE "Warning|warn\(" | tee "$RESULTS/const_max_parity.txt" | tail -25
fi

# ═══ [1+2] CLEAN (compile/load + warm runs, no profiler) ═══
echo "=== [1+2] CLEAN: NUM_RUNS=$CLEAN_RUNS, no profiler ==="
NUM_RUNS="$CLEAN_RUNS" "${LAUNCH[@]}" inference_neuron_ti2v.py > "$RESULTS/clean.log" 2>&1
echo "clean exit=$?"
grep -E "denoising:|DONE:|graph breaks|Warm run|Error|ERROR|Traceback|vae_tiled|mp4 write|nki_conv3d|conv3d:|NRT|nrt" "$RESULTS/clean.log" \
  | grep -vE "^ *File |nrt_infodump|Failed to allocate|tensor_allocate" | tail -80
echo "device OOM lines (Failed to allocate DEVICE): $(grep -c 'Failed to allocate DEVICE' "$RESULTS/clean.log")"
mapfile -t CS < <(steps_of "$RESULTS/clean.log")
T_CLEAN=$(printf '%s\n' "${CS[@]:1}" | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}')
echo "CLEAN s/step per run: ${CS[*]:-none}  -> T_CLEAN (median of warm runs) = ${T_CLEAN:-N/A} s/step"
cp /tmp/wan2_ti2v_output.mp4 "$RESULTS/output_clean.mp4" 2>/dev/null \
  && echo "VALIDATE THE OUTPUT: $RESULTS/output_clean.mp4 ($GIT_BRANCH@${GIT_SHA:0:8})"

# NEFF cache location is image-dependent; find what this process compiled.
find /tmp /var/tmp /root -name '*.neff' -size +100k 2>/dev/null | grep -v -e /tmp/results -e /tmp/systrace \
  > "$RESULTS/neff_cache_list.txt" || true
echo "cached NEFFs (>100KB): $(wc -l < "$RESULTS/neff_cache_list.txt")"

if [ "${SKIP_PROFILE:-0}" = "1" ]; then
  echo "════════ SUMMARY (SKIP_PROFILE=1) ════════"
  echo "  provenance : $GIT_BRANCH@${GIT_SHA:0:8}  TP=$TP LNC=${LNC_CONFIG:-?}"
  echo "  CLEAN      : ${T_CLEAN:-N/A} s/step"
  echo "  video      : $RESULTS/output_clean.mp4"
  exit 0
fi

# ═══ [3a] PROFILED (runtime inspector on PROFILE_RANKS) ═══
SYSTRACE=/tmp/systrace; rm -rf "$SYSTRACE"; mkdir -p "$SYSTRACE"
echo "=== [3a] PROFILED: NUM_RUNS=1 NUM_STEPS=$PROFILE_STEPS, inspector on ranks ${PROFILE_RANKS:-0} ==="
SYSTRACE_DIR="$SYSTRACE" NUM_RUNS=1 NUM_STEPS="$PROFILE_STEPS" \
  "${LAUNCH[@]}" --no-python bash scripts/prof/rank_env.sh python3 inference_neuron_ti2v.py \
  > "$RESULTS/profiled.log" 2>&1
echo "profiled exit=$?"
grep -E "denoising:|DONE:|Error|Traceback|Failed to allocate" "$RESULTS/profiled.log" | tail -10
T_PROF=$(steps_of "$RESULTS/profiled.log" | tail -1)
echo "  trace_info.pb count: $(find "$SYSTRACE" -name trace_info.pb | wc -l)"
echo "  .ntff count        : $(find "$SYSTRACE" -name '*.ntff' | wc -l)   size $(du -sh "$SYSTRACE" | cut -f1)"

# ═══ [4] pair device traces, per-graph summaries, ranking ═══
ANA="$RESULTS/analysis"; JSON="$ANA/summary_json"; mkdir -p "$JSON"
python3 scripts/prof/pair_systrace.py "$SYSTRACE" "$ANA" 2>&1 | tee "$ANA/pairing.txt"
RANKDIR=$(cat "$ANA/rank_dir.txt" 2>/dev/null || true)

if [ -n "$RANKDIR" ]; then
  echo "=== [3b] system timeline -> perfetto ==="
  mkdir -p "$RESULTS/perfetto"
  timeout 1800 "$PROF" view -d "$RANKDIR" --output-format perfetto \
    --output-file "$RESULTS/perfetto/system_profile.pftrace" > "$ANA/systrace_perfetto.log" 2>&1 || true
  [ -s "$RESULTS/perfetto/system_profile.pftrace" ] && gzip -f "$RESULTS/perfetto/system_profile.pftrace" \
    && echo "  -> perfetto/system_profile.pftrace.gz" || { echo "  system perfetto failed:"; tail -3 "$ANA/systrace_perfetto.log"; }
  timeout 1800 "$PROF" view -d "$RANKDIR" --output-format summary-text --ignore-dma-trace \
    > "$ANA/system_summary.txt" 2>&1 || true
  echo "  system summary-text: $(wc -l < "$ANA/system_summary.txt") lines"
  echo "=== system summary (rank ${PROFILE_RANKS:-0}) ==="
  head -150 "$ANA/system_summary.txt"
fi

N_OK=0; N_TRY=0
while IFS=$'\t' read -r NEFF NTFF CNT NB; do
  [ "$N_TRY" -ge "$MAX_SUMMARY" ] && break
  N_TRY=$((N_TRY+1))
  S=$(basename "$NEFF" .neff)
  timeout 600 "$PROF" view -n "$NEFF" -s "$NTFF" --output-format summary-json \
    > "$JSON/$S.json" 2>/dev/null || true
  if [ -s "$JSON/$S.json" ] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$JSON/$S.json" 2>/dev/null; then
    N_OK=$((N_OK+1))
  else
    rm -f "$JSON/$S.json"
  fi
done < "$ANA/pairs.tsv"
echo "per-graph summary-json OK: $N_OK / $N_TRY (cap MAX_SUMMARY=$MAX_SUMMARY of $(wc -l < "$ANA/pairs.tsv") paired graphs)"

python3 scripts/prof/profile_all_neffs.py "$JSON" "$ANA/exec_counts.json" 2>&1 | tee "$RESULTS/perneff_breakdown.txt"
python3 scripts/prof/rank_neffs.py "$JSON" 2>&1 | tee "$RESULTS/neff_ranking.txt"

# Preserve the heaviest graphs (by total_time x exec_count) as pre-joined pairs.
PAIRS="$RESULTS/pairs"; mkdir -p "$PAIRS"
python3 - "$JSON" "$ANA/exec_counts.json" "$MAX_PAIRS" > "$ANA/top_pairs.txt" <<'PY'
import glob, json, os, sys
jdir, counts, n = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3])
rows = []
for f in glob.glob(os.path.join(jdir, "*.json")):
    d = json.load(open(f)); nodes = [v for v in d.values() if isinstance(v, dict)] or [d]
    nd = max(nodes, key=lambda x: x.get("total_time", 0) or 0)
    s = os.path.basename(f)[:-5]
    rows.append(((nd.get("total_time", 0) or 0) * counts.get(s, 1), s,
                 (nd.get("tensor_engine_active_time_percent", 0) or 0) * 100))
for w, s, t in sorted(rows, reverse=True)[:n]:
    print(s, f"{w*1e3:.1f}ms", f"tensor{t:.0f}")
PY
i=0
while read -r S W T; do
  i=$((i+1))
  LINE=$(awk -F'\t' -v s="$S.neff" '{n=$1; sub(".*/","",n); if (n==s) {print; exit}}' "$ANA/pairs.tsv")
  NEFF=$(echo "$LINE" | cut -f1); NTFF=$(echo "$LINE" | cut -f2)
  D="$PAIRS/$(printf 'rank%02d' $i)_${S:0:24}__${W}_${T}"
  mkdir -p "$D"; cp "$NEFF" "$D/"; cp "$NTFF" "$D/$S.ntff"; cp "$JSON/$S.json" "$D/summary.json"
  echo "preserved $D ($(wc -c < "$D/$S.ntff") bytes ntff)"
done < "$ANA/top_pairs.txt"

TOPD=$(ls -d "$PAIRS"/rank01_* 2>/dev/null | head -1 || true)
if [ -n "$TOPD" ]; then
  echo "=== dominant graph summary-text: $(basename "$TOPD") ==="
  timeout 600 "$PROF" view -n "$TOPD"/*.neff -s "$TOPD"/*.ntff --output-format summary-text \
    > "$RESULTS/dominant_summary.txt" 2>&1 || true
  grep -E "total_time|tensor_engine_active_time_percent|vector_engine_active_time_percent|scalar_engine_active_time_percent|gpsimd_engine_active_time_percent|dma_active_time_percent|cc_|collective|mfu|matmul_instruction_count|transpose_flops|software_dynamic_dma_packet_percent|hbm" \
    "$RESULTS/dominant_summary.txt" | head -60
fi

# Explorer upload bundle: the chosen rank dir's .pb files + the preserved pairs.
if [ -n "$RANKDIR" ]; then
  UB=/tmp/upload_bundle; rm -rf "$UB"; mkdir -p "$UB" "$RESULTS/explorer"
  cp "$RANKDIR"/*.pb "$UB/" 2>/dev/null || true
  for d in "$PAIRS"/rank*; do [ -d "$d" ] && cp "$d"/*.neff "$d"/*.ntff "$UB/" 2>/dev/null; done
  echo "upload bundle inventory:"; for f in "$UB"/*; do echo "  $(wc -c < "$f") $(basename "$f")"; done
  tar -czf "$RESULTS/explorer/upload_bundle.tar.gz" -C "$UB" .
fi

# ═══ [4b] --single-io sweep (capped; collective graphs deadlock -> expect 0/N for the DiT) ═══
echo "=== [4b] --single-io sweep over the $SWEEP_MAX largest cached NEFFs ==="
SW="$ANA/single_io"; mkdir -p "$SW"; OK=0; N=0
while read -r NEFF; do
  N=$((N+1)); h=$(basename "$NEFF" .neff)
  NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 timeout 120 "$PROF" capture --single-io -n "$NEFF" -s "$SW/$h.ntff" \
    > "$SW/$h.capture.log" 2>&1 || continue
  [ -f "$SW/$h.ntff" ] && timeout 300 "$PROF" view -n "$NEFF" -s "$SW/$h.ntff" --output-format summary-json \
    > "$SW/$h.json" 2>/dev/null && [ -s "$SW/$h.json" ] && OK=$((OK+1))
done < <(xargs -r ls -S < "$RESULTS/neff_cache_list.txt" 2>/dev/null | head -"$SWEEP_MAX")
echo "single-io captured+viewed OK: $OK / $N"
if [ "$OK" -gt 0 ]; then
  echo "=== [4b] single-io engine table + ranking ==="
  python3 scripts/prof/profile_all_neffs.py "$SW" 2>&1 | tee "$RESULTS/single_io_breakdown.txt"
  python3 scripts/prof/rank_neffs.py "$SW" 2>&1 | tee "$RESULTS/single_io_ranking.txt"
  # If the runtime trace gave no pairs, preserve the single-io captures (content-hash pairs).
  if ! ls -d "$PAIRS"/rank* >/dev/null 2>&1; then
    i=0
    for h in $(python3 scripts/prof/top_neffs.py "$SW" "$MAX_PAIRS"); do
      i=$((i+1)); NEFF=$(grep "/$h.neff$" "$RESULTS/neff_cache_list.txt" | head -1)
      [ -n "$NEFF" ] && [ -f "$SW/$h.ntff" ] || continue
      D="$PAIRS/$(printf 'rank%02d' $i)_${h:0:24}__singleio"
      mkdir -p "$D"; cp "$NEFF" "$D/"; cp "$SW/$h.ntff" "$D/"; cp "$SW/$h.json" "$D/summary.json"
      echo "preserved $D"
    done
  fi
fi

# ═══ [5] instruction-level op/DMA report on the preserved pairs ═══
if ls -d "$PAIRS"/rank* >/dev/null 2>&1; then
  echo "=== [5] parquet + duckdb on preserved pairs ==="
  pip install --quiet duckdb 2>/dev/null || true
  MAX_PAIRS="$MAX_PAIRS" ORDER=rank INGEST_TIMEOUT="${INGEST_TIMEOUT:-2400}" \
    bash scripts/prof/analyze_trace.sh "$PAIRS" "$ANA/trace_analysis" 2>&1 | tail -150
else
  echo "=== [5] skipped: no preserved pairs ==="
fi

echo "════════ SUMMARY ════════"
echo "  provenance : $GIT_BRANCH@${GIT_SHA:0:8}  TP=$TP LNC=${LNC_CONFIG:-?}"
echo "  CLEAN      : ${T_CLEAN:-N/A} s/step  (median of warm runs; THE latency)"
echo "  PROFILED   : ${T_PROF:-N/A} s/step  (inspector on; overhead delta only)"
echo "  video      : $RESULTS/output_clean.mp4  <- VALIDATE"
echo "  ranking    : $RESULTS/neff_ranking.txt"
echo "═════════════════════════"
