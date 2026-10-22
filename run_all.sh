#!/bin/sh
# run_all.sh -- rerun the checks in fermat-parity/code and fermat-parity/checks; the results go into the folder
# given as argument (default: rerun).
# Usage (from the top folder of the repository):   sh run_all.sh rerun
# Then:   python3 -I compare_outputs.py fermat-parity/outputs rerun/outputs
#         python3 -I compare_outputs.py fermat-parity/checks/outputs rerun/checks
# The searches of fermat-parity/certify are rerun by run_certify.sh (see README.txt, Section 2).
# The interpreter is python3, or the value of the environment variable PYTHON (for example PYTHON=python).
set -e
OUT=${1:-rerun}
PY=${PYTHON:-python3}
T=$(dirname "$0")
mkdir -p "$OUT/outputs" "$OUT/checks"
for s in parity structure_lp lifts fourprimes xi420; do
  "$PY" -I -B "$T/fermat-parity/code/$s.py" "$OUT/outputs/$s.json" > "$OUT/outputs/$s.out"
done
for s in aoki_xi check_lifts420 part2_checks; do
  "$PY" -I -B "$T/fermat-parity/checks/$s.py" > "$OUT/checks/$s.out"
done
grep -H "ALL CHECKS PASS" "$OUT"/outputs/*.out
