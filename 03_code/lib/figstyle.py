"""One style for every figure of the manuscript (2026-09-17): serif text matching the TMLR body font, small labels,
no in-figure titles (captions carry them), a colour-blind-safe palette, readable collection names."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000", "#999999"]  # Okabe-Ito
NAMES = {"antique": "ANTIQUE", "cast19": "CAsT 2019", "dbpedia-entity": "DBpedia-Entity", "dl212223": "TREC DL 21--23",
         "trec-covid": "TREC-COVID", "webis-touche2020": "Touché 2020", "nfcorpus": "NFCorpus", "scifact": "SciFact",
         "arguana": "ArguAna", "cqadupstack-android": "CQADupStack", "dl2019": "TREC DL 2019", "dl2020": "TREC DL 2020"}
JUDGES = {"llm": "Qwen3-8B", "rr": "Qwen3-Reranker", "inv": "inverted reranker", "mistral": "Mistral-7B", "mpnet": "MPNet rule"}
ARMS = {"uniform": "uniform sampling", "uniform_nz": "uniform over decision-relevant documents", "weighted": "decision-weight sampling",
        "strat_pilot": "stratified by pilot variance", "ai_calib": "active inference, calibrated-judge rule + CV",
        "ai_resid": "active inference, residual-estimated rule + CV", "ai_robust_0.3": "active inference, robust mixture + CV",
        "weighted_cv": "decision-weight sampling + judge CV ($\\lambda=1$)", "weighted_cvl": "decision-weight sampling + judge CV ($\\lambda$ fitted)",
        "active_cv": "calibrated-judge sampling + CV", "active_judge": "calibrated-judge sampling", "ai_robust_0.5": "active inference, robust mixture (0.5) + CV"}


def use():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "axes.grid": False,
    })


def name(c):
    return NAMES.get(c, c).replace("--", "–")
