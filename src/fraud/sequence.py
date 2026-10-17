"""LSTM / GRU sequence models (PyTorch).

Each sample is the window of the last k transactions ending at the current one, in time order.
The label is the fraud label of the current (last) transaction. Windows only look backwards,
so validation/test windows may contain earlier-period rows (known history) but never future rows.
On ULB the window is the global transaction stream because no card identifier exists.
"""
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


class SeqNet(nn.Module):
    def __init__(self, n_features, cell="GRU", hidden=64):
        super().__init__()
        rnn = nn.GRU if cell == "GRU" else nn.LSTM
        self.rnn = rnn(n_features, hidden, batch_first=True)
        self.static = nn.Linear(n_features, hidden)     # direct path for the current transaction
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(torch.cat([out[:, -1], self.static(x[:, -1])], dim=1)).squeeze(1)


class SequenceModel:
    def __init__(self, cell="GRU", window=10, hidden=64, epochs=8, batch=512, lr=2e-3,
                 neg_fraction=0.25, seed=42, verbose=False):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def _windows(self, X, idx):
        # X: (N, F) tensor for the whole time-ordered stream; idx: positions of current rows
        offs = torch.arange(-self.window + 1, 1)
        return X[(torch.as_tensor(idx)[:, None] + offs).clamp(min=0)]

    def _predict_idx(self, X, idx):
        self.net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                out.append(torch.sigmoid(self.net(self._windows(X, idx[s:s + 4096]))))
        return torch.cat(out).numpy()

    def fit_predict(self, stream_X, y, train_idx, val_idx, test_idx):
        """stream_X: (N, F) array of the full time-ordered stream. Returns (p_val, p_test)."""
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        scaler = StandardScaler().fit(stream_X[train_idx])          # train rows only
        X = torch.tensor(scaler.transform(stream_X), dtype=torch.float32)
        y = np.asarray(y)
        train_idx = train_idx[train_idx >= self.window - 1]
        pos, neg = train_idx[y[train_idx] == 1], train_idx[y[train_idx] == 0]
        self.net = SeqNet(X.shape[1], self.cell, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        n_neg = int(len(neg) * self.neg_fraction)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(n_neg / len(pos), 50.0)))
        best, best_state, patience = -1, None, 0
        for ep in range(self.epochs):
            self.net.train()
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]); rng.shuffle(idx)
            for s in range(0, len(idx), self.batch):
                b = idx[s:s + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(self._windows(X, b)), torch.tensor(y[b], dtype=torch.float32))
                loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            ap = average_precision_score(y[val_idx], self._predict_idx(X, val_idx))
            if self.verbose:
                print(f"    {self.cell} k={self.window} epoch {ep + 1}: val PR-AUC {ap:.4f}")
            if ap > best:
                best, patience = ap, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    break
        self.net.load_state_dict(best_state)
        return self._predict_idx(X, val_idx), self._predict_idx(X, test_idx)
