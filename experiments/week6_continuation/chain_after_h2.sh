#!/usr/bin/env bash
# Week 6 Continuation: R1 -> R2 -> R3 screen -> R3 validation -> F, after H2 has frozen Tuned.
# Stops at the first non-zero exit (any SELECTION_RULES.md STOP condition, error, or refusal),
# so the sealed FINAL block is opened only if every earlier stage completed cleanly.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=results/week6_continuation/logs
W=8

until [ -f results/week6_continuation/H2/tuned_frozen.sha256 ]; do
    if ! pgrep -f "experiments.week6_continuation.cont_h2" >/dev/null; then
        sleep 5
        [ -f results/week6_continuation/H2/tuned_frozen.sha256 ] && break
        echo "CHAIN STOP: H2 ended without freezing Tuned-CLF-DR-CBF"; exit 1
    fi
    sleep 30
done
echo "CHAIN: H2 frozen, starting R1 $(date)"

$PY -m experiments.week6_continuation.cont_r1 --workers $W > $LOG/R1.log 2>&1
echo "CHAIN: R1 done $(date)"
$PY -m experiments.week6_continuation.cont_r2 --workers $W > $LOG/R2.log 2>&1
echo "CHAIN: R2 done $(date)"
$PY -m experiments.week6_continuation.cont_r3 --phase screen --workers $W > $LOG/R3_screen.log 2>&1
echo "CHAIN: R3 screen done $(date)"
$PY -m experiments.week6_continuation.cont_r3 --phase validate --workers $W > $LOG/R3_validation.log 2>&1
echo "CHAIN: R3 validation done $(date)"
$PY -m experiments.week6_continuation.cont_final --workers $W > $LOG/F.log 2>&1
echo "CHAIN: F done $(date)"
$PY -m experiments.week6_continuation.report > $LOG/report.log 2>&1
echo "CHAIN COMPLETE $(date)"
