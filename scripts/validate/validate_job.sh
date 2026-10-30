#!/usr/bin/env bash
# Quality validation of the loop_engineering configuration against main, in one pod.
#
# Runs (each NUM_RUNS=$RUNS, default 3: compile + warm runs; the last is timed and dumped via DUMP_DIR):
#   main_code   the main branch checkout, main env              -> mp4 only (main has no DUMP_DIR)
#   off         this branch, every new flag off (= main env)    -> dump + mp4
#   exact       as off but exact SDPA on all 30 layers          -> dump (bf16 reference trajectory)
#   loop        this branch, the loop_engineering flags         -> dump + mp4
#   loop_rep    loop again (NEFFs cached)                       -> dump (run-to-run determinism)
# Then, on loop's final latent: tiled VAE decode with VAE flags off vs on, scored against an exact
# fp32 CPU decode; the step-0 noise prediction of each run against an fp32 CPU DiT on its own input;
# full-trajectory latent / per-frame PSNR comparisons; a contact sheet; a PASS/WARN verdict.
#
# Bit-level checks (LSB): mp4 main_code == off (the flags-off branch IS main), loop == loop_rep
# (deterministic), VAE flags claimed bit-identical are covered by the VAE on/off decode.
# Usage (repo root, in the job pod): bash scripts/validate/validate_job.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
OUT=/tmp/validate; rm -rf "$OUT"; mkdir -p "$OUT"; echo "$OUT" > /tmp/results_LATEST_validate
LAUNCH=(torchrun --nproc_per_node="${TP_DEGREE:-8}" --master_port=29500)
export MODEL_PATH="${MODEL_PATH:-/tmp/Wan2.2-TI2V-5B}"
THREADS="${THREADS:-$(nproc)}"
RUNS="${VALIDATE_RUNS:-3}"  # run 2 can still pay NKI kernel traces; time the last run

MAIN_ENV=(FAST_ATTN_IMPL=constmax FAST_CROSS_IMPL=sdpa CFG_PARALLEL=0 FAST_SP=0 FAST_LOCAL_CROSS=0
          FAST_A2A_O=0 FAST_NKI_MLP=0 FAST_MOD_SPLIT=0 FAST_CROSS_MERGE=0
          CFG_GROUPS=contiguous VAE_FUSE_NORM=0 VAE_STREAM=0 NKI_CONV3D_SMALL_COUT=0 VAE_SUBPIXEL=0
          VAE_FUSE_CAT=0 VAE_FUSE_RES=0 VAE_FUSE_SHORTCUT=0 VAE_TCHUNK=1 VAE_STRIPS=0 VAE_STRIP_NKI_NORM=0 VAE_SUBPIXEL_MERGE=0 VAE_STRIP_ATTN=0 VAE_CONV1X1_MM=0 VAE_LAT_ALL=0 VAE_SHIP_THREAD=0 VAE_U8_SHIP=0 DEVICE_SCHEDULER=0 MP4_VFORK=0)
EXACT_ENV=("${MAIN_ENV[@]}" FAST_ATTN_IMPL=sdpa)
LOOP_ENV=(FAST_ATTN_IMPL=cte FAST_CROSS_IMPL=cte CFG_PARALLEL=1 FAST_SP=1 FAST_LOCAL_CROSS=1
          FAST_A2A_O=1 FAST_NKI_MLP=1 FAST_MOD_SPLIT=1 FAST_CROSS_MERGE=1
          CFG_GROUPS=interleave VAE_FUSE_NORM=1 VAE_STREAM=1 NKI_CONV3D_SMALL_COUT=1 VAE_SUBPIXEL=1
          VAE_FUSE_CAT=1 VAE_FUSE_RES=1 VAE_FUSE_SHORTCUT=1 VAE_TCHUNK=4 VAE_STRIPS=1 VAE_STRIP_NKI_NORM=1 VAE_SUBPIXEL_MERGE=1 VAE_STRIP_ATTN=1 VAE_CONV1X1_MM=1 VAE_LAT_ALL=1 VAE_SHIP_THREAD=1 VAE_U8_SHIP=1 DEVICE_SCHEDULER=1 MP4_VFORK=1)
VAE_OFF=(VAE_FUSE_NORM=0 NKI_CONV3D_SMALL_COUT=0 VAE_SUBPIXEL=0 VAE_FUSE_CAT=0 VAE_FUSE_RES=0 VAE_FUSE_SHORTCUT=0 VAE_TCHUNK=1 VAE_STRIPS=0 VAE_STRIP_NKI_NORM=0 VAE_SUBPIXEL_MERGE=0 VAE_STRIP_ATTN=0 VAE_CONV1X1_MM=0 VAE_LAT_ALL=0 VAE_SHIP_THREAD=0 VAE_U8_SHIP=0)
VAE_ON=(VAE_FUSE_NORM=1 NKI_CONV3D_SMALL_COUT=1 VAE_SUBPIXEL=1 VAE_FUSE_CAT=1 VAE_FUSE_RES=1 VAE_FUSE_SHORTCUT=1 VAE_TCHUNK=4 VAE_STRIPS=1 VAE_STRIP_NKI_NORM=1 VAE_SUBPIXEL_MERGE=1 VAE_STRIP_ATTN=1 VAE_CONV1X1_MM=1 VAE_LAT_ALL=1 VAE_SHIP_THREAD=1 VAE_U8_SHIP=1)

run() {  # <name> <dir> <env...> : NUM_RUNS=$RUNS inference in <dir>, dump to $OUT/<name>
  local name="$1" dir="$2"; shift 2
  echo "=== run $name ($*) ==="
  rm -f /tmp/wan2_ti2v_output.mp4
  (cd "$dir" && env "$@" NUM_RUNS="$RUNS" DUMP_DIR="$OUT/$name" "${LAUNCH[@]}" inference_neuron_ti2v.py) \
    > "$OUT/$name.log" 2>&1
  echo "  exit=$?"; grep -E "Run [0-9]+/$RUNS DONE" "$OUT/$name.log" | sed 's/.*═══ /    /; s/ ═══//'
  grep -m3 -E "Traceback|Error" "$OUT/$name.log" | cut -c1-200
  mkdir -p "$OUT/$name"; cp /tmp/wan2_ti2v_output.mp4 "$OUT/$name/output.mp4" 2>/dev/null || true
}

# ── main checkout (same commit the branch forked from is fine; take main's tip) ──
MAIN_DIR=/tmp/wan2-main
rm -rf "$MAIN_DIR"; git clone -q -b main "$(git remote get-url origin)" "$MAIN_DIR" \
  && echo "main checkout: $(git -C "$MAIN_DIR" log --oneline -1)"
[ -d "$MAIN_DIR" ] && run main_code "$MAIN_DIR" "${MAIN_ENV[@]}"
run off   "$REPO" "${MAIN_ENV[@]}"
run exact "$REPO" "${EXACT_ENV[@]}"
run loop  "$REPO" "${LOOP_ENV[@]}"
run loop_rep "$REPO" "${LOOP_ENV[@]}"

echo "=== bit-level (LSB) checks ==="
same() { cmp -s "$1" "$2" && echo "IDENTICAL" || echo "DIFFERENT"; }
B_MAIN=$(same "$OUT/main_code/output.mp4" "$OUT/off/output.mp4")
echo "  mp4 main_code vs off (flags-off branch == main): $B_MAIN"
B_DET=$(python3 - "$OUT/loop" "$OUT/loop_rep" <<'PY'
import sys, numpy as np, torch
a, b = sys.argv[1:]
lat = torch.equal(torch.load(f"{a}/latent.pt"), torch.load(f"{b}/latent.pt"))
fr = np.array_equal(np.load(f"{a}/frames.npy"), np.load(f"{b}/frames.npy"))
print("IDENTICAL" if lat and fr else f"DIFFERENT (latent equal {lat}, frames equal {fr})")
PY
)
echo "  loop vs loop_rep (determinism, latent + frames): $B_DET"
echo "  mp4 loop vs loop_rep: $(same "$OUT/loop/output.mp4" "$OUT/loop_rep/output.mp4")"

echo "=== full trajectory: final latent + frames vs exact (bf16 sdpa) ==="
python3 scripts/validate/compare.py "$OUT/exact" "$OUT/off" "$OUT/loop" 2>&1 | tee "$OUT/compare_vs_exact.txt"
echo "=== full trajectory: loop vs off (= main) ==="
python3 scripts/validate/compare.py "$OUT/off" "$OUT/loop" 2>&1 | tee "$OUT/compare_vs_off.txt"

echo "=== VAE isolated: same latent (loop's), VAE flags off (tiles) vs on (strips) ==="
for v in off on; do
  if [ "$v" = off ]; then E=("${VAE_OFF[@]}"); else E=("${VAE_ON[@]}"); fi
  env "${E[@]}" REPS="$RUNS" "${LAUNCH[@]}" scripts/validate/vae_tiled_check.py "$OUT/loop" "$OUT/vae_$v" \
    > "$OUT/vae_$v.log" 2>&1; echo "  vae_$v exit=$?"; grep -E 'rep [0-9]+:' "$OUT/vae_$v.log" | sed 's/^/    /'
done
echo "  VAE on vs off frames: $(python3 -c "import numpy as np,sys; a,b=(np.load(f'$OUT/vae_{x}/frames.npy') for x in ('off','on')); print('IDENTICAL' if np.array_equal(a,b) else 'DIFFERENT, max |diff| %d' % np.abs(a.astype(int)-b).max())" 2>&1)"
echo "=== VAE vs exact fp32 CPU decode (untiled) ==="
USE_NKI_VAE=0 USE_NKI_CONV3D=0 VAE_COMPILE=0 VAE_FUSE_NORM=0 VAE_FUSE_CAT=0 VAE_FUSE_RES=0 VAE_FUSE_SHORTCUT=0 VAE_TCHUNK=1 VAE_STRIPS=0 VAE_SUBPIXEL=0 VAE_STREAM=0 THREADS="$THREADS" python3 scripts/validate/vae_cpu_ref.py "$OUT/loop" "$OUT/vae_fp32" \
  "$OUT/vae_off/frames.npy" "$OUT/vae_on/frames.npy" 2>&1 | grep -E "vae_cpu_ref|Error" | tee "$OUT/vae_fp32.txt"

echo "=== step-0 noise prediction vs fp32 CPU DiT (each on its own input) ==="
THREADS="$THREADS" python3 scripts/validate/ref_fp32.py "$OUT/off" "$OUT/ref_off" "$OUT/exact" 2>&1 \
  | grep -E "rel_rms|Error" | tee "$OUT/step0_off.txt"
THREADS="$THREADS" python3 scripts/validate/ref_fp32.py "$OUT/loop" "$OUT/ref_loop" 2>&1 \
  | grep -E "rel_rms|Error" | tee "$OUT/step0_loop.txt"

python3 scripts/validate/contact_sheet.py "$OUT/contact_sheet.png" 0,20,50,80 \
  "$OUT/exact" "$OUT/off" "$OUT/loop" 2>&1 | tail -1

# ── verdict ──
S_OFF=$(grep -E "^ *off:" "$OUT/step0_off.txt" | grep -oE "[0-9.]+$")
S_EX=$(grep -E "^ *exact:" "$OUT/step0_off.txt" | grep -oE "[0-9.]+$")
S_LOOP=$(grep -E "^ *loop:" "$OUT/step0_loop.txt" | grep -oE "[0-9.]+$")
P_OFF=$(grep -E "^ *off:" "$OUT/compare_vs_exact.txt" | grep -oE "video PSNR +[0-9.]+" | grep -oE "[0-9.]+$")
P_LOOP=$(grep -E "^ *loop:" "$OUT/compare_vs_exact.txt" | grep -oE "video PSNR +[0-9.]+" | grep -oE "[0-9.]+$")
W_LOOP=$(grep -E "^ *loop:" "$OUT/compare_vs_exact.txt" | grep -oE "worst frame +[0-9.]+" | grep -oE "[0-9.]+$")
V_OFF=$(grep "vae_off/frames" "$OUT/vae_fp32.txt" | grep -oE "PSNR [0-9.]+" | grep -oE "[0-9.]+$")
V_ON=$(grep "vae_on/frames" "$OUT/vae_fp32.txt" | grep -oE "PSNR [0-9.]+" | grep -oE "[0-9.]+$")
T_OFF=$(grep -oE "Run $RUNS/$RUNS DONE.*total=[0-9.]+" "$OUT/off.log" | grep -oE "[0-9.]+$")
T_LOOP=$(grep -oE "Run $RUNS/$RUNS DONE.*total=[0-9.]+" "$OUT/loop.log" | grep -oE "[0-9.]+$")
python3 - "$S_OFF" "$S_EX" "$S_LOOP" "$P_OFF" "$P_LOOP" "$W_LOOP" "$V_OFF" "$V_ON" "$T_OFF" "$T_LOOP" "$B_MAIN" "$B_DET" <<'PY' | tee "$OUT/VERDICT.txt"
import sys
f = lambda s: float(s) if s not in ("", None) else None
s_off, s_ex, s_loop, p_off, p_loop, w_loop, v_off, v_on, t_off, t_loop = map(f, sys.argv[1:11])
b_main, b_det = sys.argv[11:13]
rows = []
def chk(name, ok, detail):
    rows.append(("PASS" if ok else ("WARN" if ok is not None else "N/A "), name, detail))
chk("total time", None if None in (t_off, t_loop) else t_loop < t_off, f"loop {t_loop}s vs main-env {t_off}s (A100 16.92s)")
chk("step-0 vs fp32", None if None in (s_loop, s_ex) else s_loop <= max(s_ex, s_off or 0) * 1.05,
    f"loop {s_loop} | main-env {s_off} | exact sdpa {s_ex}  (rel-RMS, bf16 floor ~0.048)")
chk("video vs exact traj", None if None in (p_loop, p_off) else p_loop >= p_off - 1.0,
    f"loop {p_loop} dB vs main-env {p_off} dB (PSNR vs exact-sdpa run; worst loop frame {w_loop} dB)")
chk("VAE vs fp32 decode", None if None in (v_on, v_off) else v_on >= v_off - 0.1,
    f"VAE flags on {v_on} dB vs off {v_off} dB")
chk("main == flags-off (mp4 bytes)", b_main == "IDENTICAL", b_main)
chk("deterministic (loop x2)", b_det == "IDENTICAL", b_det)
print("════════ VALIDATION VERDICT ════════")
for st, n, d in rows:
    print(f"  {st}  {n:30} {d}")
print("  contact sheet: /tmp/validate/contact_sheet.png (rows exact / main-env / loop; frames 0,20,50,80)")
PY
tar -czf "$OUT.tar.gz" --exclude='*.pt' -C /tmp validate 2>/dev/null
echo "ARCHIVE: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
