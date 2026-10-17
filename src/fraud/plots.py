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
