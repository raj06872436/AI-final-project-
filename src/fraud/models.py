"""Model factory. Scaling and resampling sit inside pipelines so they are fit on training data only."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _xgb(seed, spw=1.0):
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=spw,
                         eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed)


def _rf(seed, cw=None):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                  class_weight=cw, n_jobs=-1, random_state=seed)


def make_model(name: str, seed: int, pos_weight: float):
    """pos_weight = negatives / positives in the training split."""
    models = {
        "Logistic Regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "SMOTE + Logistic Regression": lambda: Pipeline([
            ("scale", StandardScaler()), ("smote", SMOTE(random_state=seed)),
            ("clf", LogisticRegression(max_iter=3000))]),
        "Random Forest": lambda: _rf(seed, "balanced_subsample"),
        "SMOTE + Random Forest": lambda: Pipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)), ("clf", _rf(seed))]),
        "XGBoost": lambda: _xgb(seed, pos_weight),
        # Large scale_pos_weight makes LightGBM diverge on this data (val PR-AUC ~0.01); settings chosen on
        # the validation period: no reweighting, stronger leaf regularization (val PR-AUC ~0.78).
        "LightGBM": lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                           min_child_samples=100, reg_lambda=5.0,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           n_jobs=-1, verbose=-1, random_state=seed),
        "MLP": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True,
                                  max_iter=200, random_state=seed))]),
    }
    return models[name]()


MODEL_NAMES = ["Logistic Regression", "SMOTE + Logistic Regression", "Random Forest",
               "SMOTE + Random Forest", "XGBoost", "LightGBM", "MLP"]


def make_imbalance_variant(strategy: str, seed: int, pos_weight: float):
    """XGBoost under different class-imbalance strategies (Section 11 of the paper)."""
    if strategy == "None":
        return _xgb(seed, 1.0)
    if strategy == "Class weighting":
        return _xgb(seed, pos_weight)
    if strategy == "SMOTE":
        return Pipeline([("smote", SMOTE(sampling_strategy=0.1, random_state=seed)), ("clf", _xgb(seed))])
    if strategy == "Undersampling":
        return Pipeline([("under", RandomUnderSampler(sampling_strategy=0.05, random_state=seed)), ("clf", _xgb(seed))])
    raise ValueError(strategy)


IMBALANCE_STRATEGIES = ["None", "Class weighting", "SMOTE", "Undersampling"]
