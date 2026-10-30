"""Novelty experiment: explanation-drift-triggered retraining under delayed labels.

    python run_novelty.py            # 3 seeds (~1 h on 4 CPU cores)
    python run_novelty.py --quick    # 1 seed
    python run_novelty.py --analyse-only   # rebuild tables, figures and report from saved CSVs
    python run_novelty.py --memory   # extension: explanation memory without retraining (~15 min)

The model is trained on the training period (hours 0-33.4). The validation + test periods (hours 33.4-48)
are then replayed as a stream of 24 chunks; each strategy scores a chunk before seeing its labels and
receives labels one chunk late. Four scenarios are replayed:

  real       the stream exactly as recorded
  harmless   from chunk 8, the 10 features the model relies on LEAST shift by +2 SD (e.g. an upstream
             logging change in fields the model ignores) - an alarm here is a false alarm
  harmful    from chunk 8, the 3 features the model relies on MOST shift by +3 SD (e.g. a feature-pipeline
             or unit change in the key fields) - the model degrades and should be retrained
  flip       from chunk 8, the sign of the 3 most important features flips - fraud becomes invisible to the
             model while the overall input distribution barely changes (a known blind spot of label-free
             monitoring, included to show the limit honestly)

Outputs: results/novelty_*.csv, results/figures/novelty_*.png, reports/NOVELTY.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fraud import adaptive, data, features, models   # noqa: E402
from fraud import config as C                        # noqa: E402
from fraud import plots                              # noqa: E402

T0 = time.time()
N_CHUNKS, DELAY, PERIOD, ONSET = 24, 1, 4, 8
PROPOSED = "Explanation-drift trigger (proposed)"
HYBRID = "Explanation drift + label guard (proposed)"
STRATEGIES = {
    "Frozen (train once)": dict(strategy="frozen"),
    "Periodic (every chunk)": dict(strategy="periodic", period=1),
    f"Periodic (every {PERIOD} chunks)": dict(strategy="periodic", period=PERIOD),
    "Input-drift trigger (PSI)": dict(strategy="input_drift"),
    PROPOSED: dict(strategy="explanation_drift"),
    HYBRID: dict(strategy="hybrid"),
}
SCENARIOS = ("real", "harmless", "harmful", "flip")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt.get(c, lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))(r[c])
                                   for c in cols) + " |\n" for _, r in df.iterrows())
    return head + body


def make_stream(name, Xs, order, sd):
    if name == "real":
        return Xs
    if name == "harmless":
        return adaptive.inject_shift(Xs, N_CHUNKS, ONSET, order[-10:], 2.0, sd)
    if name == "harmful":
        return adaptive.inject_shift(Xs, N_CHUNKS, ONSET, order[:3], 3.0, sd)
    X = Xs.copy()
    start = np.array_split(np.arange(len(X)), N_CHUNKS)[ONSET][0]
    X[start:, order[:3]] *= -1
    return X


def load_stream(seed):
    """Training history, replayed stream, and the feature ranking that decides which features get shifted."""
    df = features.build_features(data.clean(data.load_raw()))
    train, val, test = data.chronological_split(df)
    feats = C.STATIC_FEATURES + C.TEMPORAL_FEATURES
    stream = pd.concat([val, test])
    Xh, yh = train[feats].to_numpy(np.float32), train[C.TARGET].to_numpy()
    Xs, ys = stream[feats].to_numpy(np.float32), stream[C.TARGET].to_numpy()
    pos_w = (yh == 0).sum() / yh.sum()
    # Feature importance of the first-seed model on recent training data defines which features are shifted
    base = models.make_model("XGBoost", seed, pos_w).fit(Xh, yh)
    order = np.argsort(-np.abs(adaptive.shap_matrix(base, Xh[-20000:])).mean(0))
    return dict(feats=feats, Xh=Xh, yh=yh, Xs=Xs, ys=ys, pos_w=pos_w, base=base, order=order, sd=Xh.std(0),
                chunks=np.array_split(np.arange(len(ys)), N_CHUNKS))


def main(quick=False):
    seeds = C.SEEDS[:1] if quick else C.SEEDS
    d = load_stream(seeds[0])
    Xh, yh, ys, pos_w, chunks = d["Xh"], d["yh"], d["ys"], d["pos_w"], d["chunks"]
    post = np.arange(chunks[ONSET][0], len(ys))
    log(f"stream: {len(ys):,} rows, {ys.sum()} frauds, {N_CHUNKS} chunks of ~{len(chunks[0]):,}, label delay {DELAY}")
    log(f"most relied-on: {[d['feats'][i] for i in d['order'][:3]]}; "
        f"least: {[d['feats'][i] for i in d['order'][-10:]]}")

    rows, per_chunk = [], []
    for scen in SCENARIOS:
        Xsc = make_stream(scen, d["Xs"], d["order"], d["sd"])
        for seed in seeds:
            make = lambda: models.make_model("XGBoost", seed, pos_w)
            for name, kw in STRATEGIES.items():
                pc, sc, n_rt, rt_s = adaptive.run_stream(make, Xh, yh, Xsc, ys, N_CHUNKS, DELAY, **kw)
                rows.append(result_row(scen, name, seed, pc, sc, ys, post, n_rt, rt_s))
                per_chunk.append(pc.assign(scenario=scen, strategy=name, seed=seed))
                log(f"  {scen:8s} seed {seed:4d} {name:38s} retrains {n_rt:2d}  PR-AUC {rows[-1]['pr_auc']:.4f}  "
                    f"after onset {rows[-1]['post_onset_pr_auc']:.4f}")
            # Control: as many retrains as the proposed trigger, at random times
            n_prop = [r for r in rows if r["scenario"] == scen and r["seed"] == seed
                      and r["strategy"] == PROPOSED][0]["retrains"]
            rng = np.random.default_rng(seed)
            for draw in range(3 if n_prop else 0):
                at = sorted(rng.choice(N_CHUNKS - 1, n_prop, replace=False).tolist())
                pc, sc, n_rt, rt_s = adaptive.run_stream(make, Xh, yh, Xsc, ys, N_CHUNKS, DELAY,
                                                         strategy="scheduled", retrain_at=at)
                rows.append(result_row(scen, "Random times (same no. of retrains)", seed, pc, sc, ys, post, n_rt, rt_s))
            if n_prop:
                log(f"  {scen:8s} seed {seed:4d} random-time control: {n_prop} retrains x 3 draws")

    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "novelty_runs.csv", index=False)
    pd.concat(per_chunk).to_csv(C.RESULTS_DIR / "novelty_chunks.csv", index=False)
    analyse(d, seeds, runtime=(time.time() - T0) / 60)


MEMORY = "Explanation memory, no retraining (extension)"


def run_memory(quick=False):
    """Extension: after a SHAP-drift or label-guard alarm, add a kNN score over recently labelled cases in SHAP
    space instead of retraining. Writes results/novelty_memory.csv and appends a section to reports/NOVELTY.md."""
    seeds = C.SEEDS[:1] if quick else C.SEEDS
    d = load_stream(seeds[0])
    ys, post = d["ys"], np.arange(d["chunks"][ONSET][0], len(d["ys"]))
    rows = []
    for scen in SCENARIOS:
        Xsc = make_stream(scen, d["Xs"], d["order"], d["sd"])
        for seed in seeds:
            make = lambda: models.make_model("XGBoost", seed, d["pos_w"])
            pc, sc, n_rt, rt_s = adaptive.run_stream(make, d["Xh"], d["yh"], Xsc, ys, N_CHUNKS, DELAY,
                                                     strategy="hybrid", memory=True, retrain=False)
            rows.append(result_row(scen, MEMORY, seed, pc, sc, ys, post, n_rt, rt_s))
            log(f"  {scen:8s} seed {seed:4d} memory  PR-AUC after onset {rows[-1]['post_onset_pr_auc']:.4f}")
    mem = pd.DataFrame(rows)
    mem.to_csv(C.RESULTS_DIR / "novelty_memory.csv", index=False)
    g = mem.groupby("scenario", sort=False)
    t = g[["retrains", "post_onset_pr_auc"]].mean().join(g.post_onset_pr_auc.std(ddof=0).rename("post_onset_pr_auc_std"))
    comp = pd.read_csv(C.RESULTS_DIR / "novelty_comparison.csv").set_index(["scenario", "strategy"]).post_onset_pr_auc
    t["frozen"] = [comp[(s, "Frozen (train once)")] for s in t.index]
    t["proposed_with_retraining"] = [comp[(s, HYBRID)] for s in t.index]
    report = C.ROOT / "reports" / "NOVELTY.md"
    text = report.read_text().split("\n## 4. Extension")[0]
    report.write_text(text + "\n## 4. Extension: explanation memory (no retraining)\n\n"
                      "After an alarm, each new transaction also gets q = share of confirmed frauds among its 10 nearest "
                      "labelled neighbours in SHAP space (stream chunks labelled since the alarm), and the score becomes "
                      "1 - (1 - p)(1 - q). New labels change the score at once, without retraining. "
                      f"Seeds {list(seeds)}; PR-AUC from the shift onset.\n\n"
                      + md_table(t.reset_index(), ["scenario", "retrains", "post_onset_pr_auc", "post_onset_pr_auc_std",
                                                   "frozen", "proposed_with_retraining"]))
    print(t.to_string())


def analyse(d=None, seeds=None, runtime=None):
    """Tables, detector behaviour, alarm attribution, figures and report from results/novelty_{runs,chunks}.csv."""
    runs = pd.read_csv(C.RESULTS_DIR / "novelty_runs.csv")
    chunks_df = pd.read_csv(C.RESULTS_DIR / "novelty_chunks.csv")
    seeds = seeds or sorted(runs.seed.unique().tolist(), key=list(C.SEEDS).index)
    d = d or load_stream(seeds[0])
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["retrains", "retrain_seconds", "pr_auc", "post_onset_pr_auc", "recall_at_budget",
            "post_onset_recall_at_budget", "frauds_caught_at_budget"]
    g = runs.groupby(["scenario", "strategy"], sort=False)[cols]
    table = g.mean().join(g.std(ddof=0)[["post_onset_pr_auc"]].add_suffix("_std")).reset_index()
    table.to_csv(C.RESULTS_DIR / "novelty_comparison.csv", index=False)

    # Detector behaviour on the frozen model: false alarms before onset, first alarm after onset
    det = []
    for scen in SCENARIOS:
        fz = chunks_df[(chunks_df.scenario == scen) & (chunks_df.strategy.str.startswith("Frozen"))
                       & (chunks_df.seed == seeds[0])]
        for sig, label in (("explanation", "Explanation drift (SHAP)"), ("input", "Input drift (PSI)")):
            a = fz[f"{sig}_alarm"].to_numpy()
            after = np.where(a[ONSET:])[0]
            det.append({"scenario": scen, "signal": label, "alarms_before_onset": int(a[:ONSET].sum()),
                        "alarms_after_onset": int(a[ONSET:].sum()),
                        "detection_delay_chunks": int(after[0]) if len(after) else "never"})
    det = pd.DataFrame(det); det.to_csv(C.RESULTS_DIR / "novelty_detection.csv", index=False)

    # Explain the alarm: each feature's contribution to the explanation-drift score (frozen first-seed model,
    # first chunk after onset). Contributions sum to the chunk's drift score.
    mon = adaptive.DriftMonitor("explanation", len(d["chunks"][0])).fit(d["base"], d["Xh"])
    attr = {}
    for scen in SCENARIOS:
        cur = adaptive.shap_matrix(d["base"], make_stream(scen, d["Xs"], d["order"], d["sd"])[d["chunks"][ONSET]])
        attr[scen] = adaptive.drift_attribution(mon.ref, cur, d["feats"]) * adaptive.explanation_drift(mon.ref, cur)
    pd.DataFrame(attr).to_csv(C.RESULTS_DIR / "novelty_drift_attribution.csv")

    sel = lambda scen, strat: chunks_df[(chunks_df.scenario == scen) & (chunks_df.strategy == strat)
                                        & (chunks_df.seed == seeds[0])]
    plots.novelty_timeline({s: sel(s, "Frozen (train once)") for s in SCENARIOS},
                           {s: sel(s, HYBRID) for s in SCENARIOS}, ONSET, C.FIG_DIR / "novelty_drift_timeline.png")
    plots.novelty_tradeoff(table, C.FIG_DIR / "novelty_tradeoff.png")
    plots.novelty_attribution(attr, mon.threshold, C.FIG_DIR / "novelty_drift_attribution.png")

    path = C.RESULTS_DIR / "novelty_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    summary = {"seeds": list(seeds), "chunks": N_CHUNKS, "label_delay_chunks": DELAY, "onset_chunk": ONSET,
               "stream_rows": int(len(d["ys"])), "stream_frauds": int(d["ys"].sum()),
               "most_important": [d["feats"][i] for i in d["order"][:3]],
               "least_important": [d["feats"][i] for i in d["order"][-10:]],
               "explanation_threshold_initial": mon.threshold,
               "runtime_minutes": runtime if runtime is not None else old.get("runtime_minutes", float("nan"))}
    path.write_text(json.dumps(summary, indent=2))
    write_report(summary, table, det, pd.DataFrame(attr))
    print(table.to_string()); print(det.to_string())
    log("Done. See reports/NOVELTY.md")


def result_row(scen, name, seed, pc, sc, ys, post, n_rt, rt_s):
    s = adaptive.stream_summary(ys, sc, pc)
    s_post = adaptive.stream_summary(ys[post], sc[post], pc[pc.chunk >= ONSET])
    return {"scenario": scen, "strategy": name, "seed": seed, "retrains": n_rt, "retrain_seconds": rt_s, **s,
            "post_onset_pr_auc": s_post["pr_auc"], "post_onset_recall_at_budget": s_post["recall_at_budget"]}


def write_report(s, table, det, attr):
    f1 = {"retrains": lambda v: f"{v:.1f}", "frauds_caught_at_budget": lambda v: f"{v:.1f}",
          "retrain_seconds": lambda v: f"{v:.0f}"}
    t = ["# Novelty: explanation-drift-triggered retraining\n\n",
         "Generated by `run_novelty.py`. ",
         f"Stream = validation + test periods ({s['stream_rows']:,} transactions, {s['stream_frauds']} frauds) in "
         f"{s['chunks']} chunks; labels arrive {s['label_delay_chunks']} chunk late; XGBoost (static + temporal); "
         f"seeds {s['seeds']}; injected shifts start at chunk {s['onset_chunk']}; runtime {s['runtime_minutes']:.1f} min.\n\n",
         f"Most relied-on features (shifted in *harmful* / *flip*): {', '.join(s['most_important'])}. "
         f"Least relied-on (shifted in *harmless*): {', '.join(s['least_important'])}.\n\n",
         "## 1. Strategy comparison (mean over seeds)\n\n",
         "`pr_auc` = whole stream; `post_onset_*` = chunks from the shift onset on; `recall_at_budget` = share of "
         "frauds caught when analysts review the top 0.2 % of each chunk; `retrain_seconds` is wall-clock time "
         "and depends on machine load.\n\n",
         md_table(table, ["scenario", "strategy", "retrains", "pr_auc", "post_onset_pr_auc", "post_onset_pr_auc_std",
                          "recall_at_budget", "post_onset_recall_at_budget", "retrain_seconds"], f1),
         "\n## 2. Detector behaviour (frozen model, first seed)\n\n",
         "An alarm before the onset, or any alarm in the *harmless* scenario, is a false alarm.\n\n",
         md_table(det, ["scenario", "signal", "alarms_before_onset", "alarms_after_onset", "detection_delay_chunks"]),
         f"\n## 3. Which features moved (contribution to the explanation-drift score, chunk {s['onset_chunk']}; "
         f"alarm threshold {s['explanation_threshold_initial']:.3f})\n\n",
         md_table(attr.loc[attr.max(1).sort_values(ascending=False).index[:10]].reset_index()
                  .rename(columns={"index": "feature"}), ["feature", *SCENARIOS]),
         "\n![timeline](../results/figures/novelty_drift_timeline.png)\n",
         "![tradeoff](../results/figures/novelty_tradeoff.png)\n",
         "![attribution](../results/figures/novelty_drift_attribution.png)\n"]
    (C.ROOT / "reports" / "NOVELTY.md").write_text("".join(t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyse-only", action="store_true",
                    help="rebuild tables, figures and report from results/novelty_{runs,chunks}.csv")
    ap.add_argument("--memory", action="store_true", help="run only the explanation-memory extension")
    args = ap.parse_args()
    if args.memory:
        run_memory(args.quick)
    else:
        analyse() if args.analyse_only else main(args.quick)
