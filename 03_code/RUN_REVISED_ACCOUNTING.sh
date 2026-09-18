#!/usr/bin/env bash
# Revised-accounting re-run (decisions 2026-09-14): 81/82 v2 (shared ledger, full pilot cost), 83 v2 (exact oracle
# + plug-in design), 84 (J50 tables), 85 (Gate A / Gate B). Run from 03_code. Requires the pools bundle:
#   $POOLS/<stack>/runs/candidates/<name>.csv, <name>_meta.csv, <name>_llm.csv (+ _mistral/_mpnet where used)
#   stacks/names as in the locked runs: judged/dbpedia-entity, dlv2/dl212223, cast/cast19, antique/antique,
#   plus the training pools legacy/{nfcorpus,scifact,arguana,cqadupstack-android} (LODO relevance model).
# Every run writes a *_v2_<pilot_cost>_<bound>.csv next to the legacy file; legacy files are never overwritten.
set -euo pipefail
POOLS=${POOLS:-/workspace/pools_bundle}
TRAIN_DIR=${TRAIN_DIR:-$POOLS/legacy/runs/candidates}
DRAWS=${DRAWS:-300}
BOUNDS=${BOUNDS:-"t bet"}
LOG=${LOG:-/workspace/run_logs}; mkdir -p "$LOG"
cd "$(dirname "$0")"

# stack  names           judges          budgets
CFG=(
  "judged  dbpedia-entity llm,rr          20 30 45 60 90"
  "dlv2    dl212223       llm             20 30 45 60 90"
  "cast    cast19         llm,rr,inv      30 45 60 90"
  "antique antique        llm,rr,inv      30 60 90 120 150"
)

run() {  # $1 script, rest args; runs in background, one log per job
  local name; name=$(echo "$*" | tr ' /' '__' | cut -c1-120)
  echo "[start] $*"; nohup python3 "$@" > "$LOG/$name.log" 2>&1 &
}

for line in "${CFG[@]}"; do
  read -r stack names judges budgets <<<"$line"
  IFS=, read -ra J <<<"$judges"
  for judge in "${J[@]}"; do
    for bound in $BOUNDS; do
      run 81_sampling_baselines.py --pools "$POOLS" --stack "$stack" --names "$names" --judge "$judge" --train_dir "$TRAIN_DIR" --budget_full $budgets --draws "$DRAWS" --pilot_cost full --sampling shared --bound "$bound"
      run 82_active_inference.py   --pools "$POOLS" --stack "$stack" --names "$names" --judge "$judge" --train_dir "$TRAIN_DIR" --budget_full $budgets --draws "$DRAWS" --pilot_cost full --bound "$bound"
      if [ "$judge" = llm ]; then
        run 83_menu_allocation.py  --pools "$POOLS" --stack "$stack" --names "$names" --judge "$judge" --train_dir "$TRAIN_DIR" --budget_full $budgets --draws "$DRAWS" --pilot_cost full --bound "$bound"
      fi
    done
  done
done
wait
echo "[all runs finished] -> unify + gates"
python3 84_unify_metrics.py
for bound in $BOUNDS; do
  python3 85_gates.py --eps 0.02 --bound "$bound" --baseline static_sum   | tee "$LOG/gates_eps0.02_${bound}_static_sum.txt"
  python3 85_gates.py --eps 0.02 --bound "$bound" --baseline static_sum_v | tee "$LOG/gates_eps0.02_${bound}_static_sum_v.txt"
  python3 85_gates.py --eps 0.01 --bound "$bound" --baseline static_sum   | tee "$LOG/gates_eps0.01_${bound}_static_sum.txt"
done
