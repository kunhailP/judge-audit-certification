"""Shared human-label ledger (review 2026-09-14, item 3).

Every human relevance label requested during one audit run is recorded once per (query, document). The cost of a
run is the number of *unique* labels requested — pilot labels, labels shared by several policy comparisons and
labels re-requested by a later round all count exactly once. This replaces the per-comparison averaging
(`nlab += samp.sum()/len(others)` in 82) and the cutoff-only pilot cost (`cost_full[tr].sum()`) used before.

Usage
    L = Ledger()
    L.request(qid, doc_indices)         # returns nothing; marks (qid, d) as labelled
    L.request_full(qid, pool_size)      # a fully labelled query (pilot / query-level audit)
    L.cost                              # unique (qid, d) pairs so far
    L.cost_by_query[qid]                # per-query unique labels
The ledger never touches the label values: estimators keep reading the true labels from the pool arrays; the
ledger only decides what is *charged*.
"""
from collections import defaultdict
import numpy as np


class Ledger:
    __slots__ = ("_seen", "cost_by_query", "n_requests")

    def __init__(self):
        self._seen = {}                       # qid -> boolean mask of labelled docs
        self.cost_by_query = defaultdict(int)
        self.n_requests = 0                   # total requests incl. duplicates (for the overlap diagnostic)

    def request(self, qid, docs, pool_size=None):
        docs = np.asarray(docs, dtype=int)
        if docs.size == 0:
            return
        m = self._seen.get(qid)
        if m is None:
            n = int(pool_size) if pool_size is not None else int(docs.max()) + 1
            m = np.zeros(n, bool); self._seen[qid] = m
        elif docs.max() >= len(m):
            m2 = np.zeros(int(docs.max()) + 1, bool); m2[:len(m)] = m; m = m2; self._seen[qid] = m
        new = ~m[docs]
        self.n_requests += int(docs.size)
        if new.any():
            m[docs[new]] = True
            self.cost_by_query[qid] += int(np.unique(docs[new]).size)

    def request_mask(self, qid, mask):
        self.request(qid, np.flatnonzero(np.asarray(mask, bool)), pool_size=len(mask))

    def request_full(self, qid, pool_size):
        self.request(qid, np.arange(int(pool_size)), pool_size=pool_size)

    @property
    def cost(self):
        return int(sum(self.cost_by_query.values()))

    def labelled(self, qid, pool_size):
        m = self._seen.get(qid)
        out = np.zeros(int(pool_size), bool)
        if m is not None:
            out[:min(len(m), len(out))] = m[:len(out)]
        return out

    def duplicate_fraction(self):
        """Share of requests that were repeats — the amount by which per-comparison accounting overstates cost."""
        return 1.0 - self.cost / self.n_requests if self.n_requests else 0.0


def _selftest():
    L = Ledger()
    L.request("q1", [0, 3, 5], pool_size=10); assert L.cost == 3
    L.request("q1", [3, 5, 7]); assert L.cost == 4 and L.n_requests == 6
    L.request_full("q2", 8); assert L.cost == 12
    L.request("q2", [1, 2]); assert L.cost == 12
    assert abs(L.duplicate_fraction() - (1 - 12 / 16)) < 1e-12
    # the reviewer's example: two comparisons, 20 docs, pi = 0.2 each, independent draws
    rng = np.random.default_rng(0); tot = []; uniq = []
    for _ in range(20000):
        M = Ledger(); s = 0
        for _cmp in range(2):
            d = np.flatnonzero(rng.random(20) < 0.2); M.request("q", d, pool_size=20); s += len(d)
        tot.append(s); uniq.append(M.cost)
    print(f"[INFO] two comparisons, 20 docs, pi=0.2: per-comparison mean {np.mean(tot)/2:.2f}, "
          f"sum {np.mean(tot):.2f}, unique {np.mean(uniq):.2f} (expected 4 / 8 / 7.2)")
    assert abs(np.mean(uniq) - 7.2) < 0.1
    print("[PASS] ledger")


if __name__ == "__main__":
    _selftest()
