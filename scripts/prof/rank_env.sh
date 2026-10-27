#!/usr/bin/env bash
# torchrun --no-python wrapper for the Neuron runtime inspector.
#   SYSTEM profile (trace_info.pb, cheap) on the ranks in SYSTEM_RANKS (comma list of LOCAL_RANK,
#     or "all"; default all) -> the system timeline covers every rank.
#   DEVICE profile (.ntff, ~1.3 GB HBM) only on PROFILE_RANKS (default "0"). An SPMD model runs
#     the same graphs on every rank, so one rank's device trace is the device trace.
#   DEVICE_PROFILE_MODE: "session" (default) = all device activity per core in one .ntff;
#     "1"/"model" = first execution of each NEFF, but only for SYNC executions -- TorchNeuron
#     executes async (nrta_execute_schedule), so model mode yields ~1 tiny .ntff.
# Usage: torchrun --no-python ... bash scripts/prof/rank_env.sh python3 script.py
set -u
in_list() { [[ "$1" == "all" || ",$1," == *",${LOCAL_RANK:-0},"* ]]; }
unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_SYSTEM_PROFILE \
      NEURON_RT_INSPECT_DEVICE_PROFILE NEURON_RT_INSPECT_OUTPUT_DIR
if in_list "${SYSTEM_RANKS:-all}" || in_list "${PROFILE_RANKS:-0}"; then
  export NEURON_RT_INSPECT_ENABLE=1
  export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
  export NEURON_RT_INSPECT_OUTPUT_DIR="${SYSTRACE_DIR:?SYSTRACE_DIR not set}"
  export NEURON_RT_INSPECT_DEVICE_PROFILE=0
  if in_list "${PROFILE_RANKS:-0}"; then
    export NEURON_RT_INSPECT_DEVICE_PROFILE="${DEVICE_PROFILE_MODE:-session}"
  fi
fi
exec "$@"
