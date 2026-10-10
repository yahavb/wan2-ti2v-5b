#!/usr/bin/env bash
# Decode preserved NEFF+NTFF pairs into a TIME SERIES. Runs on ANY Linux host with the Neuron SDK.
#
# NO NEURON DEVICE IS NEEDED: the profiler only parses a recorded trace. So this runs on a trn
# instance, a plain Linux box with the SDK installed, or in a pod with no device claim -- and it
# does NOT queue behind or steal cores from a benchmark.
#
# It will NOT run on macOS: the SDK tools package has no build for Darwin (verified by resolving
# it against the vendor index). The query side is only duckdb and runs anywhere, which is why the
# output is plain text you can read off the host.
#
# Usage:
#   analyze_trace.sh <pairs-dir-or-results-tar> [outdir]
#   MAX_PAIRS=5 analyze_trace.sh ./results              # cap the work (default 5)
#   ORDER=rank analyze_trace.sh ./results               # by measured time, not by size
#   HEAD=200 BUCKETS=200 analyze_trace.sh ./results/pairs
#
# The single implementation: analyze-job.yaml calls this same script rather than repeating it.
set -uo pipefail

SRC="${1:?usage: analyze_trace.sh <pairs-dir-or-results-tar> [outdir]}"
OUT="${2:-./trace_analysis}"
MAX_PAIRS="${MAX_PAIRS:-5}"        # how many pairs to ingest at most
ORDER="${ORDER:-size}"            # size | rank  -- which ones those are
HEAD="${HEAD:-60}"
BUCKETS="${BUCKETS:-80}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-5400}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"
REPORT="$OUT/timeline_report.txt"; : > "$REPORT"

say() { echo "$@" | tee -a "$REPORT"; }

# ── the profiler CLI. neuron-profile is deprecated on newer SDKs and its capture fails with a
# banner pointing at neuron-explorer, so prefer that and fall back. This is the CAPTURE/VIEW
# subcommand, NOT the view SERVER (--ingest-only, :3001/:3002), which hangs on big DMA graphs.
PROF="$(command -v neuron-explorer || command -v neuron-profile \
        || echo /opt/aws/neuron/bin/neuron-explorer)"
if [ ! -x "$PROF" ] && ! command -v "$PROF" >/dev/null 2>&1; then
  echo "NO PROFILER CLI FOUND. This needs the Neuron SDK tools (neuron-explorer or"
  echo "neuron-profile) on PATH or at /opt/aws/neuron/bin. It is Linux-only; on macOS there is"
  echo "no build and the decode cannot be done locally at all."
  exit 1
fi
say "profiler CLI : $PROF"
say "version      : $("$PROF" --version 2>&1 | head -1)"

python3 -c "import duckdb" 2>/dev/null || {
  echo "installing duckdb (query side)"
  pip install --quiet duckdb 2>/dev/null || pip3 install --quiet duckdb 2>/dev/null || {
    python3 -m venv "$OUT/venv" && "$OUT/venv/bin/pip" install --quiet duckdb
    PY="$OUT/venv/bin/python"; }
}
PY="${PY:-python3}"
say "python       : $PY  (duckdb $("$PY" -c 'import duckdb;print(duckdb.__version__)' 2>&1))"

# ── locate the pairs. Accept a directory, or a results tar (what the benchmark leaves behind).
WORK="$OUT/work"; rm -rf "$WORK"; mkdir -p "$WORK"
if [ -f "$SRC" ] && case "$SRC" in *.tar.gz) true;; *) false;; esac; then
  say "restoring from tar: $SRC ($(du -h "$SRC" | cut -f1))"
  tar -xzf "$SRC" -C "$WORK"
  ROOT="$WORK"
else
  ROOT="$SRC"
fi
mapfile -t PAIRS < <(find "$ROOT" -type d -name 'rank*' | sort)
if [ "${#PAIRS[@]}" -eq 0 ]; then
  say "no rank*/ pair directories under $ROOT"
  exit 1
fi
# If the fixability ranking travelled with the results, print its head: it is the ordering that
# should drive which pair to open, and showing it lets the choice be checked rather than trusted.
RANKF=$(find "$ROOT" -name 'neff_ranking.txt' | head -1)
if [ -n "${RANKF:-}" ]; then
  say ""
  say "fixability ranking (from the benchmark -- the ordering that SHOULD drive the choice):"
  head -12 "$RANKF" | sed 's/^/  /' | tee -a "$REPORT" >/dev/null
  head -12 "$RANKF" | sed 's/^/  /'
fi

say ""
say "pairs found (processed in rank order):"
for p in "${PAIRS[@]}"; do say "  $(basename "$p")  [$(ls "$p" | tr '\n' ' ')]"; done

# ── ingest one pair to parquet.
# The CLI writes its tables and then IDLES on a server it never needs, so it must be backgrounded
# and killed once the output stops growing. Kill on SIZE-STABILISE, never on file existence: the
# summary table is created early and filled last, so "the file appeared" means nothing.
ingest() {
  local neff="$1" ntff="$2" out="$3"
  rm -rf "$out"; mkdir -p "$out"
  timeout "$INGEST_TIMEOUT" "$PROF" view -n "$neff" -s "$ntff" \
      --output-format parquet --output-file "$out" \
      --ignore-instruction-hierarchy --ignore-event-trace >"$out/.ingest.log" 2>&1 &
  # STABILITY IS ONLY MEANINGFUL ONCE REAL DATA IS ON DISK. The converter writes all 47 table
  # HEADERS immediately -- about 20 KB -- then parses for a long time before any rows land. An
  # earlier version read "size unchanged for 30 s" as finished, killed it during that parse, and
  # left 47 truncated files that every reader rejected with InvalidInputException. So ignore
  # stability until the output passes a floor well above header-only, and require a much longer
  # quiet period. du -sk not -sb: -b is GNU-only.
  local pid=$! prev=-1 cur stable=0 i
  local MIN_BYTES="${INGEST_MIN_BYTES:-2000000}"
  local NEED_STABLE="${INGEST_STABLE_CHECKS:-36}"
  for i in $(seq 1 1080); do
    kill -0 "$pid" 2>/dev/null || break
    cur=$(du -sk "$out" 2>/dev/null | cut -f1); cur=$(( ${cur:-0} * 1024 ))
    if [ "$cur" -ge "$MIN_BYTES" ] && [ "$cur" = "$prev" ]; then
      stable=$((stable+1)); [ "$stable" -ge "$NEED_STABLE" ] && break
    else stable=0; fi
    prev="$cur"; sleep 5
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  local n; n=$(ls "$out"/*.parquet 2>/dev/null | wc -l)
  say "    parquet tables: $n   size $(du -sh "$out" 2>/dev/null | cut -f1)"
  # PRESENCE IS NOT SUCCESS: 47 header-only files counted as a win once and the report that
  # followed was built on nothing. A reader has to be able to open a table AND find rows in it.
  # PIPESTATUS, not the pipeline status: piping to tee would make this return tee's 0 and every
  # failed ingest would read as a success -- the exact silent-success this check exists to stop.
  "$PY" "$HERE/pq_check.py" "$out" 2>&1 | tee -a "$REPORT"
  return "${PIPESTATUS[0]}"
}

# ── SELECT which pairs to ingest, and CAP it. Ingest cost scales with trace size and a big one
# can take the better part of an hour, so with many pairs an uncapped run never finishes.
#
# ORDER=size (default): biggest traces first -- what you asked for, and usually where the
#   interesting graph is. Note the cost side: the biggest traces are also the SLOWEST to ingest,
#   so this ordering front-loads the expensive ones.
# ORDER=rank: the order the benchmark assigned by MEASURED TIME (rank01 first). Prefer this when
#   the question is "what dominates" rather than "what is biggest", because size is how large the
#   compiled graph is and is not itself a cost. neff_ranking.txt is better still -- it ranks by
#   wasted time with the engine-mix gate, and exists because rank-1 by wall time is usually an
#   honest matmul and NOT the thing to fix.
# Size comes from `wc -c` on the .ntff, NOT `du -sb`: -b is GNU-only, and where it is missing du
# emits nothing, every size sorts equal, and the selection silently becomes arbitrary -- a dry run
# on a BSD userland picked the five SMALLEST traces while reporting "ordered by size". wc -c is
# portable and stats rather than reads (10 ms on a 1.1 GB file, measured). The .ntff is the right
# thing to measure because it is the trace, and ingest cost tracks it.
pair_bytes() {
  local d="$1" f n
  f=$(ls "$d"/*.ntff 2>/dev/null | head -1)
  [ -n "$f" ] || { echo 0; return; }
  n=$(wc -c < "$f" 2>/dev/null | tr -d ' ')
  case "$n" in ''|*[!0-9]*) echo 0 ;; *) echo "$n" ;; esac
}
if [ "$ORDER" = "size" ]; then
  mapfile -t SEL < <(for d in "${PAIRS[@]}"; do
      echo "$(pair_bytes "$d") $d"; done | sort -rn | cut -d' ' -f2-)
  # If every size came back 0 the ordering is meaningless -- say so and fall back rather than
  # analysing an arbitrary five while claiming they are the biggest.
  if [ "$(pair_bytes "${SEL[0]}")" = "0" ]; then
    say "WARNING: could not size any .ntff -- falling back to ORDER=rank"
    mapfile -t SEL < <(printf '%s\n' "${PAIRS[@]}")
  fi
else
  mapfile -t SEL < <(printf '%s\n' "${PAIRS[@]}")     # already name-sorted: rank01 first
fi
TOTAL=${#SEL[@]}
if [ "$MAX_PAIRS" -gt 0 ] && [ "$TOTAL" -gt "$MAX_PAIRS" ]; then
  SKIPPED=$((TOTAL - MAX_PAIRS))
  SEL=("${SEL[@]:0:$MAX_PAIRS}")
else
  SKIPPED=0
fi
say ""
say "selecting $((TOTAL - SKIPPED)) of $TOTAL pairs, ordered by $ORDER (MAX_PAIRS=$MAX_PAIRS):"
for p in "${SEL[@]}"; do
  say "  ANALYZE  $(basename "$p")  ntff $(( $(pair_bytes "$p") / 1048576 )) MB"
done
# NO SILENT CAPS: what was left out is stated, so a partial run cannot read as full coverage.
if [ "$SKIPPED" -gt 0 ]; then
  say "  NOT ANALYZED: $SKIPPED pair(s) -- raise MAX_PAIRS to include them:"
  for d in "${PAIRS[@]}"; do
    inc=0; for k in "${SEL[@]}"; do [ "$k" = "$d" ] && inc=1; done
    [ "$inc" = "0" ] && say "    skipped  $(basename "$d")  ntff $(( $(pair_bytes "$d") / 1048576 )) MB"
  done
fi

for p in "${SEL[@]}"; do
  neff=$(ls "$p"/*.neff 2>/dev/null | head -1)
  ntff=$(ls "$p"/*.ntff 2>/dev/null | head -1)
  say ""
  say "########## $(basename "$p") ##########"
  if [ -z "$neff" ] || [ -z "$ntff" ]; then
    say "  incomplete pair (need one .neff AND one .ntff) -- skip"
    continue
  fi
  say "  neff $(du -h "$neff" | cut -f1)   ntff $(du -h "$ntff" | cut -f1)"
  h=$(basename "$neff" .neff); pq="$OUT/pq_$h"

  # ── Try to put THIS PAIR's timeline into Perfetto.
  # system_profile.pftrace comes from the runtime system-trace DIRECTORY and has no connection to
  # these pairs: it shows WHEN things ran, not what ran inside a graph. Converting a pair directly
  # would put the instruction timeline in the viewer, which is the thing worth looking at.
  # Whether `view -n/-s` accepts the perfetto format for a PAIR is NOT established -- only
  # summary-json, summary-text and parquet are confirmed. So attempt it and report the outcome
  # instead of claiming the artifact exists.
  pft="$OUT/${h}.pftrace"
  if timeout 1800 "$PROF" view -n "$neff" -s "$ntff" \
        --output-format perfetto --output-file "$pft" >"$OUT/.pft_$h.log" 2>&1 && [ -s "$pft" ]; then
    gzip -f "$pft"
    say "  PAIR TIMELINE -> $(basename "$pft").gz ($(du -h "$pft.gz" | cut -f1)) -- open in ui.perfetto.dev"
  else
    # some builds ignore --output-file and drop the file beside the inputs
    alt=$(find "$(dirname "$neff")" "$OUT" -maxdepth 1 -name '*.pftrace' 2>/dev/null | head -1)
    if [ -n "${alt:-}" ] && [ -s "$alt" ]; then
      gzip -f "$alt"; say "  PAIR TIMELINE -> ${alt}.gz -- open in ui.perfetto.dev"
    else
      say "  no per-pair pftrace: this build does not emit perfetto from a neff+ntff pair. Reason:"
      tail -4 "$OUT/.pft_$h.log" 2>/dev/null | sed 's/^/    /' | tee -a "$REPORT"
      say "  the instruction detail is still produced as tables below"
    fi
  fi

  if ingest "$neff" "$ntff" "$pq"; then
    "$PY" "$HERE/pq_timeline.py" "$pq" "$(basename "$p")" \
        --head "$HEAD" --buckets "$BUCKETS" 2>&1 | tee -a "$REPORT"
    # ── WHAT RAN INSIDE THE GRAPH, in ui.perfetto.dev.
    # The runtime's system trace converts to Perfetto directly but shows WHEN graphs ran, not what
    # ran inside one. Try the tool's own pair->perfetto conversion first; only summary and table
    # formats are confirmed for a pair, so if it declines, build the trace from the instruction
    # table instead -- deterministic, since the conversion is ours: one slice per instruction, one
    # track per engine, source location attached to each slice.
    PFT="$OUT/${h}_pair.pftrace"
    if timeout 1800 "$PROF" view -n "$neff" -s "$ntff" --output-format perfetto \
          --output-file "$PFT" >"$OUT/.pft_$h.log" 2>&1 && [ -s "$PFT" ]; then
      gzip -f "$PFT"
      say "  PERFETTO (tool) -> ${PFT}.gz  -- open at https://ui.perfetto.dev"
    else
      say "  tool declined perfetto for a pair; building it from the instruction table:"
      "$PY" "$HERE/pq_to_perfetto.py" "$pq" "$OUT/${h}_pair.json.gz" \
          --scale "${PFT_SCALE:-0.001}" 2>&1 | tee -a "$REPORT"
    fi
    [ -f "$HERE/pq_dma_report.py" ] && "$PY" "$HERE/pq_dma_report.py" "$pq" "$h" 2>&1 | tee -a "$REPORT"
    [ -f "$HERE/pq_pad_shapes.py" ] && "$PY" "$HERE/pq_pad_shapes.py" "$pq" "$h" 2>&1 | tee -a "$REPORT"
  else
    say "  INGEST PRODUCED NO TABLES -- tail of the converter's own log:"
    tail -12 "$pq/.ingest.log" 2>/dev/null | sed 's/^/    /' | tee -a "$REPORT"
  fi
done

say ""
say "report: $REPORT"
