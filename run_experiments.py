"""Run every experiment from the research paper end to end on the ULB dataset.

    python run_experiments.py            # full run (3 seeds, all experiments)
    python run_experiments.py --quick    # 1 seed, shorter sequence training

Outputs go to results/ (CSV tables, figures, summary.json) and reports/RESULTS.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fraud import config as C                                         # noqa: E402
from fraud import data, evaluate, explain, features, models, plots    # noqa: E402
from fraud.sequence import SequenceModel                              # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fit_eval(name, make, train, val, test, feats, seed):
    """Fit on train, choose the threshold on validation, evaluate the test set once."""
    pos_w = (train[C.TARGET] == 0).sum() / train[C.TARGET].sum()
    m = make(name, seed, pos_w)
    m.fit(train[feats], train[C.TARGET])
    p_val = m.predict_proba(val[feats])[:, 1]
    p_test = m.predict_proba(test[feats])[:, 1]
    th = evaluate.threshold_max_f1(val[C.TARGET], p_val)
    res = evaluate.metrics(test[C.TARGET], p_test, th)
    res["val_pr_auc"] = float(evaluate.metrics(val[C.TARGET], p_val, th)["pr_auc"])
    return m, p_val, p_test, res


def repeat(name, make, train, val, test, feats, seeds, tag):
    rows, keep = [], None
    for s in seeds:
        t = time.time()
        m, pv, pt, r = fit_eval(name, make, train, val, test, feats, s)
        rows.append({"model": name, "setting": tag, "seed": s, "fit_seconds": time.time() - t, **r})
        if keep is None:
            keep = (m, pv, pt)
    log(f"  {name:28s} [{tag}] PR-AUC {np.mean([r['pr_auc'] for r in rows]):.4f}  "
        f"ROC-AUC {np.mean([r['roc_auc'] for r in rows]):.4f}  F1 {np.mean([r['f1'] for r in rows]):.4f}")
    return rows, keep


def agg(df, keys):
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "val_pr_auc"]
    g = df.groupby(keys, sort=False)[[c for c in cols if c in df]]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).fillna(0).add_suffix("_std"))
    cm = df.groupby(keys, sort=False)[["tp", "fp", "fn", "tn", "threshold"]].first()
    return out.join(cm).reset_index()


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt.get(c, lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))(r[c])
                                   for c in cols) + " |\n" for _, r in df.iterrows())
    return head + body


def main(quick=False):
    seeds = C.SEEDS[:1] if quick else C.SEEDS
    C.RESULTS_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True); C.MODEL_DIR.mkdir(exist_ok=True)
    summary = {"seeds": list(seeds), "quick": quick}

    # ---------------- 1. Dataset audit and cleaning ----------------
    log("Loading and auditing dataset")
    raw = data.load_raw()
    summary["audit_raw"] = data.audit(raw)
    df = data.clean(raw)
    summary["audit_clean"] = data.audit(df)
    plots.class_balance(summary["audit_raw"], C.FIG_DIR / "class_balance.png")
    plots.fraud_over_time(df, C.FIG_DIR / "fraud_over_time.png")
    log(f"  raw {summary['audit_raw']['rows']:,} rows / {summary['audit_raw']['fraud']} fraud; "
        f"removed {summary['audit_raw']['duplicate_rows']} duplicates "
        f"({summary['audit_raw']['duplicate_fraud_rows']} fraud) -> {len(df):,} rows")

    # ---------------- 2. Features and chronological split ----------------
    log("Building static + past-only temporal features")
    df = features.build_features(df)
    train, val, test = data.chronological_split(df)
    splits = data.split_summary((train, val, test))
    splits.to_csv(C.RESULTS_DIR / "splits.csv", index=False)
    summary["splits"] = splits.to_dict("records")
    assert train.Time.max() <= val.Time.min() <= val.Time.max() <= test.Time.min(), "time leakage"
    S, ST = C.STATIC_FEATURES, C.STATIC_FEATURES + C.TEMPORAL_FEATURES
    y_test = test[C.TARGET].to_numpy()

    # ---------------- 3. Baselines on static features (RQ1/RQ2) ----------------
    log("Experiment 1: baseline models, static features, chronological split")
    rows, probs, fitted = [], {}, {}
    for name in models.MODEL_NAMES:
        r, (m, pv, pt) = repeat(name, models.make_model, train, val, test, S, seeds, "static")
        rows += r; probs[name] = pt; fitted[name] = (m, pv, pt)
    base = pd.DataFrame(rows); base.to_csv(C.RESULTS_DIR / "baselines_runs.csv", index=False)
    base_agg = agg(base, ["model"]); base_agg.to_csv(C.RESULTS_DIR / "baselines.csv", index=False)

    # ---------------- 4. Static vs temporal ablation (H1) ----------------
    log("Experiment 2: static vs static+temporal ablation")
    rows, temporal_probs = [], {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
        rows += [r for r in base.to_dict("records") if r["model"] == name]
        r, (m, pv, pt) = repeat(name, models.make_model, train, val, test, ST, seeds, "static+temporal")
        rows += r; temporal_probs[name] = pt; fitted[name + " (+temporal)"] = (m, pv, pt)
    abl = agg(pd.DataFrame(rows), ["model", "setting"]); abl.to_csv(C.RESULTS_DIR / "ablation_temporal.csv", index=False)
    plots.bar(abl, "model", "pr_auc_mean", "setting", "H1: static vs static + temporal features (test PR-AUC)",
              C.FIG_DIR / "ablation_temporal.png", "PR-AUC")

    # ---------------- 5. Random vs chronological split (H3) ----------------
    log("Experiment 3: random vs chronological split")
    rtr, rva, rte = data.random_split(df)
    rows = []
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        rows += [{**r, "setting": "chronological"} for r in base.to_dict("records") if r["model"] == name]
        r, _ = repeat(name, models.make_model, rtr, rva, rte, S, seeds, "random")
        rows += r
    split_cmp = agg(pd.DataFrame(rows), ["model", "setting"]); split_cmp.to_csv(C.RESULTS_DIR / "split_comparison.csv", index=False)
    plots.bar(split_cmp, "model", "pr_auc_mean", "setting", "H3: random split vs chronological split (test PR-AUC)",
              C.FIG_DIR / "split_comparison.png", "PR-AUC")

    # ---------------- 6. Class-imbalance strategies ----------------
    log("Experiment 4: class-imbalance strategies (XGBoost, static+temporal)")
    rows = []
    for strat in models.IMBALANCE_STRATEGIES:
        r, _ = repeat(strat, lambda n, s, w: models.make_imbalance_variant(n, s, w), train, val, test, ST, seeds, "XGBoost")
        rows += r
    imb = agg(pd.DataFrame(rows), ["model"]).rename(columns={"model": "strategy"})
    imb.to_csv(C.RESULTS_DIR / "imbalance_strategies.csv", index=False)

    # ---------------- 7. Sequence models + history window (RQ4, RQ7) ----------------
    log("Experiment 5: LSTM / GRU over the last k transactions")
    stream = df[ST].to_numpy(dtype=np.float32)
    yall = df[C.TARGET].to_numpy()
    n_tr, n_va = len(train), len(val)
    tr_idx, va_idx, te_idx = np.arange(n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, len(df))
    rows, seq_probs = [], {}
    windows = (5, 10, 20) if quick else (5, 10, 20, 50)
    for cell in ("LSTM", "GRU"):
        for k in windows:
            t = time.time()
            sm = SequenceModel(cell=cell, window=k, epochs=5 if quick else 8, seed=C.SEED)
            pv, pt = sm.fit_predict(stream, yall, tr_idx, va_idx, te_idx)
            th = evaluate.threshold_max_f1(yall[va_idx], pv)
            r = evaluate.metrics(y_test, pt, th)
            r["val_pr_auc"] = evaluate.metrics(yall[va_idx], pv, th)["pr_auc"]
            rows.append({"model": cell, "window": k, "fit_seconds": time.time() - t, **r})
            seq_probs[(cell, k)] = (pv, pt)
            log(f"  {cell} k={k:<3d} PR-AUC {r['pr_auc']:.4f} ROC-AUC {r['roc_auc']:.4f} F1 {r['f1']:.4f}")
    seq = pd.DataFrame(rows); seq.to_csv(C.RESULTS_DIR / "sequence_models.csv", index=False)
    plots.line(seq, "window", "pr_auc", "model", "History-window experiment: test PR-AUC vs window length k",
               C.FIG_DIR / "history_window.png")
    for cell in ("LSTM", "GRU"):                                  # best window chosen on validation
        best = seq[seq.model == cell].sort_values("val_pr_auc").iloc[-1]
        pv, pt = seq_probs[(cell, int(best.window))]
        fitted[f"{cell} (k={int(best.window)})"] = (None, pv, pt)

    # ---------------- 8. Overall comparison and final model selection ----------------
    log("Comparing all models; selecting final model by VALIDATION PR-AUC")
    allrows = []
    for name, (m, pv, pt) in fitted.items():
        vy = val[C.TARGET].to_numpy()
        th = evaluate.threshold_max_f1(vy, pv)
        allrows.append({"model": name, "val_pr_auc": evaluate.metrics(vy, pv, th)["pr_auc"], **evaluate.metrics(y_test, pt, th)})
    overall = pd.DataFrame(allrows).sort_values("val_pr_auc", ascending=False)
    overall.to_csv(C.RESULTS_DIR / "all_models.csv", index=False)
    final_name = overall.iloc[0].model
    log(f"  selected: {final_name}")
    summary["final_model"] = final_name
    _, pv_f, pt_f = fitted[final_name]

    curve_set = {k: fitted[k][2] for k in ["Logistic Regression", "Random Forest", "SMOTE + Random Forest", "MLP",
                                           "XGBoost", "XGBoost (+temporal)", "LightGBM (+temporal)"]}
    curve_set.update({k: v[2] for k, v in fitted.items() if k.startswith(("LSTM", "GRU"))})
    plots.pr_roc_curves(y_test, curve_set, C.FIG_DIR / "pr_roc_curves.png")
    plots.bar(overall.assign(metric="test PR-AUC"), "model", "pr_auc", "metric", "Test PR-AUC, all models (threshold-free)",
              C.FIG_DIR / "all_models_pr_auc.png", "PR-AUC")
    plots.confusion(overall.iloc[0], f"{final_name} (test, max-F1 threshold)", C.FIG_DIR / "confusion_final.png")

    # ---------------- 9. Operating thresholds (H5) ----------------
    log("Experiment 6: operational thresholds")
    ops = pd.DataFrame(evaluate.operating_points(val[C.TARGET].to_numpy(), pv_f, y_test, pt_f))
    ops.to_csv(C.RESULTS_DIR / "operating_points.csv", index=False)

    # ---------------- 10. Temporal drift (RQ6) ----------------
    log("Experiment 7: temporal drift across early / middle / late test windows")
    rows = []
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (+temporal)", final_name]:
        m, pv, pt = fitted[name]
        th = evaluate.threshold_max_f1(val[C.TARGET], pv)
        rows += [{"model": name, **w} for w in evaluate.window_metrics(y_test, pt, th)]
    drift = pd.DataFrame(rows).drop_duplicates(["model", "window"]); drift.to_csv(C.RESULTS_DIR / "drift.csv", index=False)
    plots.line(drift, "window", "pr_auc", "model", "Concept drift: PR-AUC by test window", C.FIG_DIR / "drift.png")

    # ---------------- 11. Explainability ----------------
    log("Experiment 8: SHAP explanations for XGBoost (+temporal)")
    xgb_t = fitted["XGBoost (+temporal)"][0]
    imp = explain.shap_analysis(xgb_t, test[ST], y_test, C.FIG_DIR)
    imp.to_csv(C.RESULTS_DIR / "shap_importance.csv", index=False)
    summary["top_shap_features"] = imp.feature.head(10).tolist()

    # Save the deployable model + demo data for the Streamlit app
    th_x = evaluate.threshold_max_f1(val[C.TARGET], fitted["XGBoost (+temporal)"][1])
    joblib.dump({"model": xgb_t, "features": ST, "threshold": th_x}, C.MODEL_DIR / "xgb_temporal.joblib")
    demo = pd.concat([test[test[C.TARGET] == 1], test[test[C.TARGET] == 0].sample(400, random_state=C.SEED)])
    demo[["Time", "Amount", C.TARGET] + ST].to_csv(C.RESULTS_DIR / "demo_transactions.csv", index=False)

    summary["runtime_minutes"] = (time.time() - T0) / 60
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, base_agg, abl, split_cmp, imb, seq, overall, ops, drift, imp)
    log("Done. See reports/RESULTS.md")


def write_report(s, base, abl, split_cmp, imb, seq, overall, ops, drift, imp):
    pm = lambda m, sd: f"{m:.4f} ± {sd:.4f}"
    a, c = s["audit_raw"], s["audit_clean"]
    t = ["# Experiment Results\n",
         f"Generated by `run_experiments.py` (seeds: {s['seeds']}, runtime {s['runtime_minutes']:.1f} min). "
         "All test numbers use a threshold chosen on the validation period; the test period is evaluated once.\n",
         "## 1. Dataset audit\n",
         f"- Raw: {a['rows']:,} transactions, {a['fraud']} fraud ({a['fraud_rate']:.3%}), {a['missing_values']} missing values, "
         f"{a['time_span_hours']:.1f} hours.\n- Removed {a['duplicate_rows']:,} exact duplicates ({a['duplicate_fraud_rows']} fraud) "
         f"-> {c['rows']:,} rows, {c['fraud']} fraud.\n- Predicting 'legitimate' for everything gives "
         f"{a['trivial_all_legit_accuracy']:.4%} accuracy and 0 recall, so accuracy is not used for selection.\n",
         "\n### Chronological split (60/20/20)\n",
         md_table(pd.DataFrame(s["splits"]), ["split", "rows", "fraud", "fraud_rate", "time_start_h", "time_end_h"],
                  {"fraud_rate": lambda v: f"{v:.3%}", "time_start_h": lambda v: f"{v:.1f}", "time_end_h": lambda v: f"{v:.1f}"}),
         "\n## 2. Baseline models (static features, chronological split, mean ± std over seeds)\n",
         "Metric columns are means over seeds; tp / fn / fp are from the first seed.\n\n",
         md_table(base.assign(**{"PR-AUC": [pm(m, d) for m, d in zip(base.pr_auc_mean, base.pr_auc_std)],
                                 "ROC-AUC": [pm(m, d) for m, d in zip(base.roc_auc_mean, base.roc_auc_std)]}),
                  ["model", "PR-AUC", "ROC-AUC", "precision_mean", "recall_mean", "f1_mean", "tp", "fn", "fp"]),
         "\n## 3. H1 - Static vs static + temporal features\n",
         md_table(abl, ["model", "setting", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "recall_mean", "precision_mean", "f1_mean"]),
         "\n## 4. H3 - Random vs chronological split\n",
         md_table(split_cmp, ["model", "setting", "pr_auc_mean", "roc_auc_mean", "recall_mean", "f1_mean"]),
         "\n## 5. Class-imbalance strategies (XGBoost, static + temporal)\n",
         md_table(imb, ["strategy", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "precision_mean", "recall_mean", "f1_mean"]),
         "\n## 6. Sequence models and history window (LSTM / GRU over the last k transactions)\n",
         md_table(seq, ["model", "window", "val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1"]),
         "\n## 7. All models (ranked by validation PR-AUC)\n",
         "Single run (first seed) of every model, including the best LSTM/GRU window chosen on validation.\n\n",
         md_table(overall, ["model", "val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1", "tp", "fn", "fp"]),
         f"\nFinal model selected on validation: **{s['final_model']}**.\n",
         "\n## 8. H5 - Operating thresholds for the final model\n",
         md_table(ops, ["rule", "threshold", "precision", "recall", "f1", "tp", "fn", "fp", "cost"],
                  {"cost": lambda v: f"{v:,.0f}", "threshold": lambda v: f"{v:.4f}"}),
         "\n## 9. Concept drift across the test period\n",
         md_table(drift, ["model", "window", "fraud", "pr_auc", "roc_auc", "recall", "precision"]),
         "\n## 10. Top SHAP features (XGBoost + temporal)\n",
         md_table(imp.head(10), ["feature", "mean_abs_shap"]),
         "\n## Figures\n",
         *[f"![{p.stem}](../results/figures/{p.name})\n" for p in sorted(C.FIG_DIR.glob("*.png"))]]
    (C.ROOT / "reports" / "RESULTS.md").write_text("".join(t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
