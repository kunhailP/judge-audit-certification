# Pre-registered predictions for the pilot-size sweep at ε = 0.03 (2026-09-15)

<!-- Extracted on 2026-09-18 from §14 of the internal review-response document of 2026-09-14 (English translation); the predictions P1–P3 below were written before any ε = 0.03 result was looked at, and the outcome paragraph was appended after the run. Cited by Appendix D of the manuscript and by 88_pilot_sweep.py. -->

Track A -- pilot sweep and fixed-candidate decomposition (2026-09-15)

Runs: `83_menu_allocation.py` with the population estimand (v3b tags), 300 draws, arms `per_pair static_sum
oracle_exact plugin_exact`, pilot fully charged, t bound. (a) `--n_train 10/40/80` (20 = existing v3b) with the
candidate chosen on the pilot and shared by all arms inside a draw; (b) `--fixed_cand best` at pilot 20 (every draw
certifies the true best policy: regret 0, so the candidate-selection effect is removed). Aggregated by
`88_pilot_sweep.py` -> `05_results/unified/PILOT_SWEEP_v3.{csv,md}`.

Findings (eps = 0.02 unless stated; eps = 0.01 is qualitatively the same):
1. Candidate quality rises with the pilot: P(regret <= eps) = 0.71-0.87 (pilot 10), 0.85-0.92 (20), 0.93-0.98 (40),
   0.99-1.00 (80). The pilot is the only place where a *wrong* candidate can enter.
2. Given a feasible candidate, the shared arms certify essentially always on ANTIQUE, CAsT and DL (>= 0.95 at every
   pilot size). DBpedia plateaus at 0.85-0.95 even when the candidate is feasible, but reaches 0.97-0.99 with the
   true best candidate -> the DBpedia residual is a *slack* problem (feasible candidates with regret close to eps),
   not a variance problem. `per_pair` (no label reuse) stays far below on every collection (CAsT <= 0.12).
3. Total cost (pilot included) is *minimised by small pilots*: pilot 10 or 20 is cheapest for every shared arm on
   every collection; pilot 40 costs 1.3-1.6x and pilot 80 costs 2.5-3.0x pilot 20. The pilot's own labels dominate
   the gain from a better candidate. Sampling labels alone (pilot excluded) do fall monotonically with pilot size.
4. Removing the candidate-selection effect (fixed best) changes J50_total by -2% (ANTIQUE, DL), -18% (DBpedia) and
   -21% (CAsT): candidate selection is a second-order cost on ANTIQUE/DL and a first-order one on CAsT/DBpedia.
5. The allocation-optimisation gain (oracle vs static_sum) stays small at every pilot size and with the fixed
   candidate (<= 14%, CAsT); a better candidate does not unmask a hidden design gain.

Pre-registered prediction for a condition not used so far (eps = 0.03, all four collections, pilot 10/20/40/80,
launched before looking at any eps = 0.03 result):
  P1  For every shared arm and collection, J50_total(pilot 40) > J50_total(pilot 20) and J50_total(pilot 80) >
      J50_total(pilot 40); pilot 10 is within +-15% of pilot 20.
  P2  ACT | feasible >= 0.95 for static_sum/oracle_exact on ANTIQUE, CAsT, DL at every pilot size.
  P3  oracle_exact saves <= 15% of J50_total relative to static_sum on every collection.
Falsification: any collection where pilot 40 beats pilot 20 in total cost (P1), or where oracle saves > 15% (P3).

Outcome of the held-out eps = 0.03 run (`88_pilot_sweep.py`, section "Held-out check"):
  P1 monotone part  : held on 12/12 (arm, collection) cells -- J50_total(40) > J50_total(20), J50_total(80) > J50_total(40).
  P1 "pilot 10 within +-15% of pilot 20": FAILED. Pilot 10 is 3-13% cheaper on CAsT and 18-30% cheaper on ANTIQUE,
      DBpedia, DL. Direction: the looser eps, the smaller the cost-minimising pilot (wrong candidates get rarer,
      the pilot's price does not). Reported as a failed prediction in Appendix D and Limitations.
  P2 held (ACT|feasible >= 0.95 on ANTIQUE, CAsT, DL for static_sum / oracle_exact at every pilot size).
  P3 held (oracle saving 0-4% of J50_total vs static_sum).
Manuscript: Appendix D gains a "Pilot size and the candidate-selection effect" paragraph + Table (tab:pilot);
Sec. 7(2)(a) and App. D(i) now attribute the DBpedia residual to small slack (fixed best -> 0.99 at eps 0.02, 0.90 at
eps 0.01) rather than to "residual variance of the hardest comparison"; Limitations records the failed sub-prediction;
App. D's closing sentence replaces "governed first by the pilot's candidate choice" with the split by collection and
the reading "diagnose the candidate's failure rate and slack before optimising where the remaining labels go".
