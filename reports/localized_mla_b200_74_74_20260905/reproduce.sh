#!/usr/bin/env bash
# Run from the repository root; pass a fresh output directory.
set -euo pipefail
out="${1:?Usage: bash reports/localized_mla_b200_74_74_20260905/reproduce.sh OUTPUT_DIR}"
export MAX_JOBS=5 FLASHINFER_NVCC_THREADS=1 OMP_NUM_THREADS=5
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python_bin="$PWD/.venv/bin/python"
# Inspect resources before JIT compilation; no concurrent GPU workloads.
nproc
free -h
uptime
df -h /tmp /workspace
nvidia-smi
if [[ -e "$out/decode/sq1/post_flops.json" || -e "$out/ncu_boundary/matrix_summary.json" ]]; then
    echo "Use a fresh output directory to preserve previous measurements." >&2
    exit 1
fi
mkdir -p "$out/validation"
"$python_bin" benchmarks/validate_cute_dsl_localized_mla_profile_target.py \
    --output-root "$out/validation" > "$out/validation/run.log" 2>&1
for sq in 1 4; do
    mkdir -p "$out/decode/sq$sq"
    "$python_bin" benchmarks/bench_cute_dsl_localized_mla.py \
        --device 0 --expected-partition-sm-counts 74 74 --seqlen-q "$sq" \
        --batch-sizes 2 4 8 16 32 64 \
        --seqlen-ks 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
        --data-initialization random --paired-warmups 20 \
        --timing-warmup-ms 500 --timing-repeat-ms 1000 \
        --timing-blocks 4 --timing-min-samples 20 \
        --output "$out/decode/sq$sq/post_flops.json" > "$out/decode/sq$sq/run.log" 2>&1
    "$python_bin" benchmarks/plot_cute_dsl_localized_mla.py --timing-only \
        --timing "$out/decode/sq$sq/post_flops.json" \
        --output-dir "$out/decode/sq$sq/figures"
done
mkdir -p "$out/ncu_boundary"
"$python_bin" benchmarks/profile_cute_dsl_localized_mla_boundary.py \
    --device 0 --expected-partition-sm-counts 74 74 \
    --timing-sources "$out/decode/sq1/post_flops.json" "$out/decode/sq4/post_flops.json" \
        reports/localized_mla_prefill_sq128_dense/iter_000_evaluation/timing.json \
    --output-root "$out/ncu_boundary" > "$out/ncu_boundary/run.log" 2>&1
# Prefill timing is intentionally reused from the completed 74/74 dense matrix.
"$python_bin" reports/localized_mla_b200_74_74_20260905/summarize.py --output-root "$out"
