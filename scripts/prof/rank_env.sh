#!/usr/bin/env bash
# torchrun --no-python wrapper: enable the Neuron runtime inspector only on the ranks in
# PROFILE_RANKS (comma list of LOCAL_RANK, default "0"). An SPMD model runs the same graphs on
# every rank, so one rank's trace is the trace, and profiling every rank multiplies trace
# size (device .ntff per graph execution) by the world size.
# Usage: torchrun --no-python ... bash scripts/prof/rank_env.sh python3 script.py
set -u
if [[ ",${PROFILE_RANKS:-0}," == *",${LOCAL_RANK:-0},"* ]]; then
  export NEURON_RT_INSPECT_ENABLE=1
  export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
  export NEURON_RT_INSPECT_DEVICE_PROFILE=1
  export NEURON_RT_INSPECT_OUTPUT_DIR="${SYSTRACE_DIR:?SYSTRACE_DIR not set}"
else
  unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_SYSTEM_PROFILE \
        NEURON_RT_INSPECT_DEVICE_PROFILE NEURON_RT_INSPECT_OUTPUT_DIR
fi
exec "$@"
