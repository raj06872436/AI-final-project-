"""Project-wide settings: paths, seeds, split ratios and feature groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

SEED = 42
SEEDS = (42, 7, 2024)            # repeated runs for mean ± std
SPLIT = (0.60, 0.20, 0.20)       # chronological train / validation / test

# Cost model for the operational threshold (H5): a missed fraud costs far more than a review.
COST_FN = 100.0
COST_FP = 5.0
REVIEW_CAPACITY = 0.002          # fraction of test transactions analysts can review

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
STATIC_FEATURES = PCA_FEATURES + ["log_amount", "hour_sin", "hour_cos"]
TEMPORAL_FEATURES = [
    "log_gap_prev", "log_gap_next_prev",
    "cnt_10s", "cnt_60s", "cnt_600s",
    "amt_mean_100", "amt_std_100", "amt_z_100",
    "amt_mean_1h", "amt_z_1h", "amt_rank_100",
]
TARGET = "Class"
