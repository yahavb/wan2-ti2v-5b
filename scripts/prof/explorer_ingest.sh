#!/usr/bin/env bash
# Start the Neuron Explorer UI and ingest everything bench_profile.sh produced, named so the
# profile list says what each one is:
#   00_SYSTEM_all_ranks              system timeline of every rank (no device data, fast)
#   01_SYSTEM_<pid>                  per-rank fallback, only if the all-ranks ingest fails
#   dev_rankNN_<component>_<ms>_...  device NEFF+NTFF pairs, heaviest first (dit/vae/t5, _cc =
#                                    has collectives)
#   02_SYSTEM_DEVICE_rank0_session   system + session device trace of the profiled rank -- the
#                                    collective DiT graphs that single-io cannot capture. Last,
#                                    because it is the slowest to ingest.
# View from your Mac: kubectl port-forward <pod> 3001:3001 3002:3002 -> http://localhost:3001
# Usage: explorer_ingest.sh <results-dir>
set -uo pipefail
RESULTS="${1:?usage: explorer_ingest.sh <results-dir>}"
PROF="$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)"
SYSTRACE=$(cat "$RESULTS/systrace_root.txt" 2>/dev/null || echo /tmp/systrace)
RANKDIR=$(cat "$RESULTS/analysis/rank_dir.txt" 2>/dev/null || true)

"$PROF" view > /tmp/explorer.log 2>&1 &
sleep 20

ingest() {  # <display-name> <timeout> <view args...>
  local name="$1" t="$2"; shift 2
  if timeout "$t" "$PROF" view "$@" --ingest-only --display-name "$name" > "/tmp/ingest_$name.log" 2>&1; then
    echo "explorer ingested: $name"
  else
    echo "explorer ingest FAILED: $name ($(tail -1 "/tmp/ingest_$name.log"))"; return 1
  fi
}

if [ -d "$SYSTRACE" ]; then
  ingest 00_SYSTEM_all_ranks 1800 -d "$SYSTRACE" --ignore-device-profile || {
    for d in $(find "$SYSTRACE" -name trace_info.pb -exec dirname {} \; | sort); do
      ingest "01_SYSTEM_$(basename "$d" | grep -oE 'pid_[0-9]+')" 1800 -d "$d" --ignore-device-profile
    done
  }
fi

for d in "$RESULTS"/pairs/rank*/; do
  [ -d "$d" ] || continue
  neff=$(ls "$d"*.neff 2>/dev/null | head -1); ntff=$(ls "$d"*.ntff 2>/dev/null | head -1)
  [ -n "$neff" ] && [ -n "$ntff" ] || continue
  ingest "dev_$(basename "$d")" 1800 -n "$neff" -s "$ntff"
done

if [ -n "$RANKDIR" ] && [ -n "$(find "$RANKDIR" -name '*.ntff' | head -1)" ]; then
  ingest 02_SYSTEM_DEVICE_rank0_session "${SESSION_INGEST_TIMEOUT:-7200}" -d "$RANKDIR"
fi
echo "EXPLORER READY: kubectl port-forward $(hostname) 3001:3001 3002:3002  -> http://localhost:3001"
