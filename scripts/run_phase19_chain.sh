#!/usr/bin/env bash
# Phase 19 chain: G -> E -> D -> F1 (PHASE19_PREREG.md order). D needs G's
# string scores; F1 needs E's L80. F2 is NOT run (needs Gowtham's approval).
set -u
cd /users/gsundar/projects/MIRA
notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/mira-g5x7k2-status" >/dev/null 2>&1 || true; }
scripts/run_phase19_block.sh G research/phase19_g_gate.py; g=$?
scripts/run_phase19_block.sh E research/phase19_e_location.py; e=$?
if [ $g -eq 0 ]; then scripts/run_phase19_block.sh D research/phase19_d_kernel.py; d=$?; else d=skipped; fi
if [ $e -eq 0 ]; then scripts/run_phase19_block.sh F1 research/phase19_f1_dimension.py; f=$?; else f=skipped; fi
notify "MIRA Phase 19 chain finished: G=$g E=$e D=$d F1=$f (F2 awaits approval)"
