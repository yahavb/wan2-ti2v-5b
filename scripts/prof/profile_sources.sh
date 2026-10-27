#!/usr/bin/env bash
# Minimal source tree for the Neuron Explorer Code Editor (Source Code Viewer). Uploading whole
# packages clogged the profiler, so only:
#   <out>/framework   the executable: inference_neuron_ti2v.py      -> --framework-source-root
#   <out>/nki         our NKI kernels (kernels/*.py, top level only) and the installed nkilib
#                     conv3d.py that the VAE actually runs            -> --nki-source-root
#   <out>.tar.gz      the same, for the web UI "Source Code" upload from your Mac
# Paths are relative to the repo / site-packages (kernels/rope.py, nkilib/experimental/conv/...).
#
# Needs the compile-time flags (job yaml): XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NKI_DEBUG_INFO=True.
# Usage: profile_sources.sh <repo> <out>
set -uo pipefail
REPO="$(cd "${1:?usage: profile_sources.sh <repo> <out>}" && pwd)"; OUT="${2:?}"
rm -rf "$OUT"; mkdir -p "$OUT/framework" "$OUT/nki/kernels"

cp "$REPO/inference_neuron_ti2v.py" "$OUT/framework/"
cp "$REPO"/kernels/*.py "$OUT/nki/kernels/"

CONV3D=$(python3 -c "import importlib.util; print(importlib.util.find_spec('nkilib.experimental.conv.conv3d').origin)" 2>/dev/null)
if [ -n "$CONV3D" ] && [ -f "$CONV3D" ]; then
  mkdir -p "$OUT/nki/nkilib/experimental/conv"; cp "$CONV3D" "$OUT/nki/nkilib/experimental/conv/"
fi

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "sources: $(cd "$OUT" && find . -name '*.py' | sort | tr '\n' ' ')-> $OUT.tar.gz"
