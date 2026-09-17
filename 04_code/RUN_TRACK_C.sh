#!/usr/bin/env bash
# Track C re-run (2026-09-17): the v3 fixed-budget grid of 81_sampling_baselines.py with the new `uniform_nz` arm and the
# per-draw records (--dump_draws: unique-label cost, certificate outcome, pilot-only variance prediction per arm), then
#   86_table2_v2.py v3   -> Table 2 with the "uniform over decision-relevant documents" row
#   94_j50_ci.py         -> paired-bootstrap intervals for J50 and for the savings ratios
#   97_pilot_rule.py     -> accuracy of the pre-audit rule (pilot-predicted variance ratio / saving vs realised)
# The other arms' draws are unchanged (uniform_nz uses its own random stream), so the existing v3 numbers reproduce exactly.
# Requires the pools bundle (see RUN_REVISED_ACCOUNTING.sh). One process per (collection, judge, budget), tag _v3b{B}.
set -euo pipefail
POOLS=${POOLS:-/workspace/pools_bundle}
TRAIN_DIR=${TRAIN_DIR:-$POOLS/legacy/runs/candidates}
DRAWS=${DRAWS:-300}
LOG=${LOG:-/workspace/run_logs}; mkdir -p "$LOG"
cd "$(dirname "$0")"

# stack   names           judges       budgets (the v3 grid of design log §13.2)
CFG=(
  "antique antique        llm,rr,inv   5 10 15 20 30 60 90 120 150"
  "cast    cast19         llm,rr,inv   15 20 30 45 60 90 120 150 200"
  "judged  dbpedia-entity llm,rr       5 10 15 20 30 45 60 90 120 150 200"
  "dlv2    dl212223       llm          5 10 15 20 30 45 60 90"
)
run() { local name; name=$(echo "$*" | tr ' /' '__' | cut -c1-120); echo "[start] $*"; nohup python3 "$@" > "$LOG/$name.log" 2>&1 & }

for line in "${CFG[@]}"; do
  read -r stack names judges budgets <<<"$line"
  IFS=, read -ra J <<<"$judges"
  for judge in "${J[@]}"; do
    for B in $budgets; do
      run 81_sampling_baselines.py --pools "$POOLS" --stack "$stack" --names "$names" --judge "$judge" --train_dir "$TRAIN_DIR" \
          --budget_full "$B" --eps 0.01 0.02 --draws "$DRAWS" --pilot_cost full --sampling shared --bound t --dump_draws --tag "_v3b${B}"
    done
  done
done
wait
echo "[81 finished] -> tables, intervals, rule"
python3 86_table2_v2.py v3 | tee "$LOG/table2_v3.txt"
python3 95_cost_model.py    | tee "$LOG/cost_model.txt"
for eps in 0.02 0.01; do
  python3 94_j50_ci.py    --eps $eps | tee "$LOG/j50_ci_eps${eps}.txt"
  python3 97_pilot_rule.py --eps $eps | tee "$LOG/pilot_rule_eps${eps}.txt"
done
