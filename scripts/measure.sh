#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dgemm"
SIZES=(512 1024 2048 4096)
IMPLS=(cublas tiled naive)

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_sweep.csv"
echo "impl,m,n,k,time_ms,gflops" > "$LOG"

for n in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "Running $impl N=$n"
    out="$("$BIN" --m "$n" --n "$n" --k "$n" --impl "$impl" --no-verify)"
    line="$(grep -m1 -E '^Impl=' <<<"$out" || true)"
    time_ms="$(sed -n 's/.*Time(ms)=\([0-9.]\+\).*/\1/p' <<<"$line")"
    gflops="$(sed -n 's/.*GFLOP\/s=\([0-9.]\+\).*/\1/p' <<<"$line")"

    if [[ -z "${time_ms}" || -z "${gflops}" ]]; then
      echo "ERROR: failed to parse output for impl=$impl n=$n" >&2
      echo "Full output:" >&2
      echo "$out" >&2
      exit 1
    fi

    echo "$impl,$n,$n,$n,$time_ms,$gflops" >> "$LOG"
  done
done

echo "Results stored in $LOG"

