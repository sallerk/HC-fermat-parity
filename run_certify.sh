#!/bin/sh
# run_certify.sh -- rerun the exhaustive searches (certify.py), their summary (summarize.py) and the z3 checks
# (z3check.py) of fermat-parity/certify, with the options of the shipped result files.
# These scripts write their results next to themselves, so this OVERWRITES the shipped results in
# fermat-parity/certify.  Run it in a copy of the repository and compare the two folders afterwards:
#   sh COPY/run_certify.sh
#   python3 -I compare_outputs.py fermat-parity/certify COPY/fermat-parity/certify
# The interpreter is python3, or the value of the environment variable PYTHON (for example PYTHON=python).
# About 1.5 hours; some runs use 8 worker processes.  certify.py stops a level at its --deadline (seconds), so
# on a slower machine a run may end with unfinished levels; its .out file then says so.
set -e
PY=${PYTHON:-python3}
C=$(dirname "$0")/fermat-parity/certify
run() { "$PY" -I -B "$C/certify.py" "$@" > /dev/null; }
run test_small --src Q --q 15,21,33 --procs 3 --selftest
run P1 --src Q --q 15,21,33,35,39,51,55,57,65,69 --procs 8 --selftest --count 102:7 --deadline 300
run S_T_upto128 --src S,T --dmax 128 --procs 8 --selftest --deadline 240
run ST_129_300 --src S,T --dmin 129 --dmax 300 --procs 8 --w2cap 4e7 --selftest --deadline 570
run ST_301_624 --src S,T --dmin 301 --dmax 700 --procs 8 --selftest --deadline 570
run Q_only_B7 --src Q --dmin 229 --dmax 300 --Bfix 7 --deadline 400
run S_195_q65_B9 --src S --q 65 --dmin 195 --dmax 195 --procs 1 --Bfix 9 --deadline 570
run S_207_q69_B9 --src S --q 69 --dmin 207 --dmax 207 --procs 1 --Bfix 9 --deadline 570
run S_220_x20_B9 --src S --q 44 --dmin 220 --dmax 220 --procs 1 --Bfix 9 --extra 20 --deadline 570
run S_255_q51_B9 --src S --q 51 --dmin 255 --dmax 255 --procs 1 --Bfix 9 --noextra --deadline 570
run S_255_x15_B9 --src S --q 51 --dmin 255 --dmax 255 --procs 1 --Bfix 9 --extra 15 --deadline 570
run S_260_q65_B9 --src S --q 65 --dmin 260 --dmax 260 --procs 1 --Bfix 9 --deadline 570
run S_276_q69_B8 --src S --q 69 --dmin 276 --dmax 276 --procs 1 --Bfix 8 --deadline 570
run S_285_q57_B8 --src S --q 57 --dmin 285 --dmax 285 --procs 1 --Bfix 8 --deadline 570
run S_308_x28_B8 --src S --q 44 --dmin 308 --dmax 308 --procs 1 --Bfix 8 --extra 28 --deadline 570
run T_182_q91_B9 --src T --q 91 --procs 1 --Bfix 9 --deadline 570
run T_276_q92_B8 --src T --q 92 --procs 1 --Bfix 8 --deadline 570
run T_exact_B9 --src T --q 93,95,111 --procs 3 --Bfix 9
"$PY" -I -B "$C/summarize.py" > /dev/null
"$PY" -I -B "$C/z3check.py" z3_P1 25 15,21,33,35,39,51,55,57,65,69 > /dev/null
"$PY" -I -B "$C/z3check.py" z3_B6_q51 280 51 6 > /dev/null
"$PY" -I -B "$C/z3check.py" z3_B6_q65 280 65 6 > /dev/null
"$PY" -I -B "$C/z3check.py" z3_B6_q69 280 69 6 > /dev/null
grep -H "jobs" "$C/summary.out" | tail -n 1
