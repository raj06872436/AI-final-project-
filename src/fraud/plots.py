"""Result figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})


def pr_roc_curves(y, probs: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p); ax[0].plot(rc, pr, label=name, lw=1.4)
        fpr, tpr, _ = roc_curve(y, p); ax[1].plot(fpr, tpr, label=name, lw=1.4)
    ax[0].axhline(np.mean(y), ls=":", c="grey", label="No-skill")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test period)")
    ax[1].plot([0, 1], [0, 1], ls=":", c="grey")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test period)", xscale="log", xlim=(1e-5, 1))
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def confusion(cm_row: dict, title, path):
    cm = np.array([[cm_row["tn"], cm_row["fp"]], [cm_row["fn"], cm_row["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def bar(df, x, y, hue, title, path, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set(title=title, ylabel=ylabel or y, xlabel="")
    ax.tick_params(axis="x", rotation=20)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def line(df, x, y, hue, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", ax=ax)
    ax.set(title=title); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def class_balance(audit: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Fraud"], [audit["legit"], audit["fraud"]], color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log"); ax.set_title(f"Class imbalance ({audit['fraud_rate']:.3%} fraud)")
    for i, v in enumerate([audit["legit"], audit["fraud"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fraud_over_time(df, path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    hours = (df.Time // 3600).astype(int)
    ax.plot(df.groupby(hours).size(), label="All transactions / hour")
    ax2 = ax.twinx(); ax2.bar(*zip(*df[df.Class == 1].groupby(hours[df.Class == 1]).size().items()), color="#C44E52", alpha=.6, label="Fraud / hour")
    ax.set(xlabel="Hour since start", ylabel="Transactions"); ax2.set_ylabel("Fraud")
    ax.set_title("Transaction volume and fraud over the 48-hour period")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)



def novelty_timeline(frozen: dict, proposed: dict, onset, path):
    """Per scenario: chunk PR-AUC (top) and label-free drift scores relative to their alarm threshold (bottom)."""
    names = list(frozen)
    fig, ax = plt.subplots(2, len(names), figsize=(4.2 * len(names), 6.5), sharex=True, sharey="row")
    for j, s in enumerate(names):
        fz, pr = frozen[s], proposed[s]
        top, bot = ax[0, j], ax[1, j]
        top.plot(fz.chunk, fz.pr_auc.rolling(3, min_periods=1).mean(), c="#8C8C8C", label="Frozen model")
        top.plot(pr.chunk, pr.pr_auc.rolling(3, min_periods=1).mean(), c="#C44E52", label="Proposed (SHAP drift + label guard)")
        for c in pr.chunk[pr.retrained_after]:
            top.axvline(c + 0.5, c="#C44E52", ls=":", lw=0.8)
        bot.plot(fz.chunk, fz.explanation_drift / fz.explanation_threshold, marker="o", ms=3, c="#C44E52",
                 label="Explanation drift (SHAP)")
        bot.plot(fz.chunk, fz.input_drift / fz.input_threshold, marker="s", ms=3, c="#4C72B0", label="Input drift (PSI)")
        bot.axhline(1, c="k", ls="--", lw=1, label="Alarm threshold")
        for a in (top, bot):
            if s != "real":
                a.axvline(onset - 0.5, c="grey", lw=1)
        top.set_title(f"{s} stream"); bot.set_xlabel("Chunk (~36 min each)")
    ax[0, 0].set_ylabel("Chunk PR-AUC (3-chunk mean)"); ax[1, 0].set_ylabel("Drift score / threshold")
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.suptitle("Label-free monitoring of the stream (dotted red = retrain fired; grey = shift onset)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def novelty_tradeoff(table, path):
    """Post-onset PR-AUC versus number of retrains, one panel per scenario."""
    scen = list(dict.fromkeys(table.scenario))
    strategies = list(dict.fromkeys(table.strategy))
    style = {st: (m, c) for st, m, c in zip(strategies, "osD^vP*X",
                                              ["#8C8C8C", "#4C72B0", "#55A868", "#8172B2", "#C44E52", "#DD8452", "#CCB974"])}
    fig, ax = plt.subplots(1, len(scen), figsize=(4.2 * len(scen), 4.3))
    for a, s in zip(np.atleast_1d(ax), scen):
        for _, r in table[table.scenario == s].iterrows():
            m, c = style[r.strategy]
            a.scatter(r.retrains, r.post_onset_pr_auc, marker=m, c=c, s=110 if "(proposed)" in r.strategy else 55,
                      edgecolors="k" if "(proposed)" in r.strategy else "none", label=r.strategy, zorder=3, alpha=.9)
        a.set(title=f"{s} stream", xlabel="Retrains", ylabel="PR-AUC after onset")
    h, l = np.atleast_1d(ax)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Detection quality vs retraining cost (up and left is better)")
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(path); plt.close(fig)


def novelty_attribution(attr: dict, threshold, path, top=8):
    """Each feature's contribution to the explanation-drift score in the first chunk after the shift."""
    names = [s for s in attr if s != "flip"]
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharex=True)
    for a, s in zip(ax, names):
        v = attr[s].head(top)[::-1]
        a.barh(v.index, v.values, color="#C44E52")
        a.set(title=f"{s} stream: score {attr[s].sum():.2f} (alarm > {threshold:.2f})",
              xlabel="Contribution to explanation-drift score")
    fig.suptitle("Explaining the drift alarm: where the model's reasoning changed")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
