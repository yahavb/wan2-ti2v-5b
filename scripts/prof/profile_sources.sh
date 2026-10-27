#!/usr/bin/env bash
# Collect every python source that issues the torch / NKI calls in the profiled graphs, for the
# Neuron Explorer Code Editor (Source Code Viewer).
#
# The NEFF debug info records ABSOLUTE paths (e.g. /usr/local/lib/python3.14/site-packages/nkilib/
# experimental/conv/conv3d.py:3459), and the docs do not say how a source root is matched, so each
# file is placed twice under the root: at its absolute path (root + /abs/path) and at its path
# relative to the repo / site-packages (root + pkg/file.py). Either rule then finds it.
#
#   <out>/framework   repo model code (wan/, models/, entry scripts) + torch_neuronx + torch/nn
#                     -> --framework-source-root
#   <out>/nki         repo kernels/ (incl. nkilib_vendor) + installed nkilib / neuronxcc.nki
#                     -> --nki-source-root
#   <out>.tar.gz      the same, for the web UI "Source Code" upload from your Mac
#
# Needs the compile-time flags to have been set (job yaml): XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1
# NKI_DEBUG_INFO=True. Without them the profile has no source locations to link.
# Usage: profile_sources.sh <repo> <out>
set -uo pipefail
REPO="$(cd "${1:?usage: profile_sources.sh <repo> <out>}" && pwd)"; OUT="${2:?}"
rm -rf "$OUT"; mkdir -p "$OUT/framework" "$OUT/nki"

put() {  # <root> <base-dir>, absolute file list on stdin: copy at absolute path + relative to base
  local root="$1" base="$2" list; list=$(mktemp)
  cat > "$list"
  (cd / && sed 's|^/||' "$list" | tar -cf - -T - 2>/dev/null) | tar -xf - -C "$root"
  (cd "$base" && sed "s|^$base/||" "$list" | tar -cf - -T - 2>/dev/null) | tar -xf - -C "$root"
  wc -l < "$list" | tr -d ' '; rm -f "$list"
}
pyfiles() { find "$@" -name '*.py' -not -path '*/__pycache__/*' 2>/dev/null; }
pkgdir() { python3 -c "import importlib.util,os,sys; s=importlib.util.find_spec(sys.argv[1]); print(os.path.dirname(s.origin))" "$1" 2>/dev/null; }

# repo: model code -> framework, kernels -> nki
pyfiles "$REPO" | grep -v -e "^$REPO/kernels/" -e "^$REPO/scripts/" -e "/\.venv/" | put "$OUT/framework" "$REPO" >/dev/null
pyfiles "$REPO/kernels" | put "$OUT/nki" "$REPO" >/dev/null

# installed packages the graphs come from
for m in torch_neuronx torch.nn nkilib neuronxcc.nki; do
  d=$(pkgdir "$m"); [ -n "$d" ] || { echo "  (not installed: $m)"; continue; }
  base=$(python3 -c "import sys; d=sys.argv[1]; n=sys.argv[2].count('.')+1; print(d.rsplit('/', n)[0])" "$d" "$m")
  case "$m" in nkilib|neuronxcc.nki) root="$OUT/nki";; *) root="$OUT/framework";; esac
  echo "  $m: $(pyfiles "$d" | put "$root" "$base") files from $d"
done

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "sources: framework $(find "$OUT/framework" -name '*.py' | wc -l) files, nki $(find "$OUT/nki" -name '*.py' | wc -l) files -> $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
