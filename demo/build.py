"""Build demo/index.html: inline engine.js, novelty.js and the novelty results from results/novelty_*.csv.

    python demo/build.py            # demo/index.html, figures referenced from ../results/figures/
    python demo/build.py --publish  # site/index.html + site/figures/, a self-contained folder to host
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STRATEGIES = {"Frozen (train once)": "frozen", "Input-drift trigger (PSI)": "psi",
              "Explanation drift + label guard (proposed)": "prop"}


def novelty_data(seed=42):
    d = pd.read_csv(RES / "novelty_chunks.csv")
    d = d[d.seed == seed]
    out = {}
    for sc in ("real", "harmless", "harmful", "flip"):
        o = {}
        for name, key in STRATEGIES.items():
            x = d[(d.scenario == sc) & (d.strategy == name)].sort_values("chunk")
            r = {"ap": [None if pd.isna(a) else round(a, 4) for a in x.pr_auc],
                 "hit": x.hits_at_budget.tolist(), "rt": x.retrained_after.astype(int).tolist()}
            if key == "frozen":
                r["xd"] = (x.explanation_drift / x.explanation_threshold).round(3).tolist()
                r["pd"] = (x.input_drift / x.input_threshold).round(3).tolist()
                r["fraud"] = x.fraud.tolist()
            if key == "prop":
                r["xa"] = x.explanation_alarm.astype(int).tolist()
                r["ga"] = x.label_guard_alarm.astype(int).tolist()
            if key == "psi":
                r["pa"] = x.input_alarm.astype(int).tolist()
            o[key] = r
        out[sc] = o
    t = pd.read_csv(RES / "novelty_comparison.csv")
    out["table"] = t[["scenario", "strategy", "retrains", "post_onset_pr_auc"]].round(4).to_dict("records")
    a = pd.read_csv(RES / "novelty_drift_attribution.csv", index_col=0)
    out["attr"] = {c: [[i, round(v, 4)] for i, v in a[c].sort_values(ascending=False).head(8).items()] for c in a}
    return json.dumps(out, separators=(",", ":"))


def main(publish=False):
    demo = ROOT / "demo"
    page = (demo / "page.src.html").read_text()
    page = page.replace("/*ENGINE*/", (demo / "engine.js").read_text())
    page = page.replace("/*NOVELTY_JS*/", (demo / "novelty.js").read_text().replace("/*NOVELTY_DATA*/", novelty_data()))
    if not publish:
        (demo / "index.html").write_text(page)
        return print("wrote demo/index.html")
    site = ROOT / "site"
    (site / "figures").mkdir(parents=True, exist_ok=True)
    for f in (RES / "figures").glob("*.png"):
        shutil.copy(f, site / "figures" / f.name)
    (site / "index.html").write_text(page.replace("../results/figures/", "figures/"))
    print("wrote site/index.html and site/figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    main(ap.parse_args().publish)
