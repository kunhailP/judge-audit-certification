#!/usr/bin/env bash
# LOCK v0.7 pipeline (NeuCLIRBench, English mono). Steps 1-3 build data and the pre-audit predictions; the predictions are
# committed BEFORE step 4 runs the audit. Run from 04_code. Requires the pools bundle (training pools) and /workspace/neuclir.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
POOLS=${POOLS:-/workspace/pools_bundle}; TRAIN_DIR=$POOLS/legacy/runs/candidates; NC=/workspace/neuclir; LOG=${LOG:-/workspace/neuclir/logs}; mkdir -p $LOG
STEP=${1:-all}; BUDGETS="2 3 5 8 10 15 20 30 45 60 90"; D=$POOLS/neuclir/runs/candidates
cd "$(dirname "$0")"
if [ "$STEP" = pool ] || [ "$STEP" = all ]; then
  python3 100_neuclir_pool.py --bench $NC/bench --texts $NC/judged_texts.tsv --out $POOLS 2>&1 | tee $LOG/pool.log
fi
if [ "$STEP" = judge ] || [ "$STEP" = all ]; then
  python3 66_llm_judge.py neuclir_eng --cand $D --texts $D/neuclir_eng_texts.tsv --queries $D/neuclir_eng_queries.tsv --batch 16 2>&1 | tee $LOG/judge.log
fi
if [ "$STEP" = predict ] || [ "$STEP" = all ]; then
  for judge in llm rr inv; do for nt in 20 10; do tag=$([ $nt = 20 ] && echo _nc || echo _nc10); for B in $BUDGETS; do
    while [ "$(jobs -rp | wc -l)" -ge 31 ]; do sleep 2; done
    nohup python3 81_sampling_baselines.py --pools $POOLS --stack neuclir --names neuclir_eng --judge $judge --train_dir $TRAIN_DIR --budget_full $B --eps 0.01 0.02 --draws 300 --n_train $nt --pilot_cost full --sampling shared --bound t --predict_only --tag ${tag}_b$B > $LOG/predict_${judge}${nt}_b$B.log 2>&1 &
  done; done; done; wait
  cd .. && for judge in llm rr inv; do python3 04_code/98_heldout_predict.py --glob "05_results/sampling_baselines/baselines_neuclir_${judge}_nc_b*_predict.csv" --tag neuclir_${judge} > $LOG/lock_${judge}.txt
    python3 04_code/98_heldout_predict.py --glob "05_results/sampling_baselines/baselines_neuclir_${judge}_nc10_b*_predict.csv" --tag neuclir_${judge}_pilot10 > $LOG/lock_${judge}_pilot10.txt; done; cd 04_code
  echo "[predict] done -> commit 05_results/heldout/PREDICTIONS_neuclir_* and the *_predict.csv files, THEN run: ./RUN_NEUCLIR.sh audit"
fi
if [ "$STEP" = audit ]; then
  for judge in llm rr inv; do for nt in 20 10; do tag=$([ $nt = 20 ] && echo _nc || echo _nc10); for B in $BUDGETS; do
    while [ "$(jobs -rp | wc -l)" -ge 31 ]; do sleep 2; done
    nohup python3 81_sampling_baselines.py --pools $POOLS --stack neuclir --names neuclir_eng --judge $judge --train_dir $TRAIN_DIR --budget_full $B --eps 0.01 0.02 --draws 300 --n_train $nt --pilot_cost full --sampling shared --bound t --dump_draws --tag ${tag}_b$B > $LOG/audit_${judge}${nt}_b$B.log 2>&1 &
  done; done; done; wait
  cd .. && for judge in llm rr inv; do for tag in nc nc10; do for eps in 0.02 0.01; do python3 04_code/97_pilot_rule.py --eps $eps --glob "05_results/sampling_baselines/baselines_neuclir_${judge}_${tag}_*_draws.csv" > $LOG/score_${judge}_${tag}_eps$eps.txt
    mv 05_results/unified/PILOT_RULE_eps$eps.csv 05_results/heldout/PILOT_RULE_neuclir_${judge}_${tag}_eps$eps.csv; mv 05_results/unified/PILOT_RULE_single_eps$eps.csv 05_results/heldout/PILOT_RULE_single_neuclir_${judge}_${tag}_eps$eps.csv 2>/dev/null || true; done; done; done
  echo "[audit] done -> scores in $LOG/score_*.txt and 05_results/heldout/"
fi
