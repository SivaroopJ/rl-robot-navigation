#!/usr/bin/env bash
# Policy matrix, remaining steps in the approved order (run after the Phase I main run and the
# development shadow run have been launched):
#   calibration (dev_B) -> smoke C -> [wait for main_B] -> main C -> latency A/B/C (1 worker, alone)
#   -> generated report tables.
# Per the user's instruction every step runs even if an earlier one fails; each exit code is logged.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=results/pursuit_policy_matrix
LOG=$R/logs
THR=$R/calibration/cd_thresholds.json

wait_for() { until [ -f "$1" ]; do sleep 30; done; }
run() { local name=$1; shift; echo "CHAIN: $name start $(date)"; "$@" > "$LOG/$name.log" 2>&1;
        local rc=$?; echo "exit=$rc" >> "$LOG/$name.log"; echo "CHAIN: $name exit=$rc $(date)"; }

wait_for $R/dev_B/records.json
run calibrate $PY -m experiments.pursuit.cd_calibrate --run $R/dev_B
run smoke_C $PY -m experiments.pursuit.policy_matrix --phase smoke --config C --thresholds $THR --workers 4
wait_for $R/main_B/records.json
run main_C $PY -m experiments.pursuit.policy_matrix --phase main --config C --thresholds $THR --workers 12
run latency_A $PY -m experiments.pursuit.policy_matrix --phase latency --config A --workers 1
run latency_B $PY -m experiments.pursuit.policy_matrix --phase latency --config B --thresholds $THR --workers 1
run latency_C $PY -m experiments.pursuit.policy_matrix --phase latency --config C --thresholds $THR --workers 1
run report $PY -m experiments.pursuit.policy_matrix_report --out MD_files/pursuit/MPC_CD_REPORT_DATA.md
echo "CHAIN COMPLETE $(date)"
