"""train_eval.py — Full benchmark: 10 x 5 repeated stratified CV,
inner 3-fold hyperparameter selection (separate from the 10% early-stopping
validation subset), outputs per-sample predictions/probabilities and the
per-fold metrics table (rebuilds Supplementary Table S1).

Usage:  python train_eval.py            # all 9 models
        python train_eval.py AM-CNN   # single model
Outputs: out/predictions.csv, out/probabilities.csv, out/per_fold_metrics.csv
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from models import build, count_params

torch.manual_seed(5000)
N_REPS, N_FOLDS, EPOCHS, PAT = 10, 5, 100, 10
GRID_SK = {
 "LR":       [{"C": c} for c in (0.01, 0.1, 1, 10)],
 "RF":       [{"n_estimators": n, "max_depth": d}
              for n in (100, 300, 500) for d in (3, 5, 10)],
 "XGBoost":  [{"n_estimators": n, "max_depth": d, "learning_rate": l}
              for n in (100, 300, 500) for d in (3, 5, 7) for l in (0.01, 0.05, 0.1)],
 "LightGBM": [{"n_estimators": n, "max_depth": d, "learning_rate": l}
              for n in (100, 300, 500) for d in (3, 5, 7) for l in (0.01, 0.05, 0.1)],
 "MLP":      [{"hidden_layer_sizes": (h,)} for h in (16, 32, 64)],
}
SK_BUILD = {
 "LR": lambda p: LogisticRegression(C=p["C"], max_iter=2000),
 "RF": lambda p: RandomForestClassifier(n_estimators=p["n_estimators"],
                                        max_depth=p["max_depth"], random_state=5000),
 "XGBoost": lambda p: XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                    learning_rate=p["learning_rate"], eval_metric="logloss",
                                    random_state=5000, verbosity=0),
 "LightGBM": lambda p: LGBMClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      learning_rate=p["learning_rate"], random_state=5000,
                                      verbose=-1),
 "MLP": lambda p: MLPClassifier(hidden_layer_sizes=p["hidden_layer_sizes"],
                                alpha=p.get("alpha", 0.001), max_iter=500, random_state=5000),
}
TORCH_MODELS = ["AM-CNN", "PM-CNN", "CNN", "Transformer", "CNN+multi-scale", "CNN+SE"]


def f1_of(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, ACC=(tp + tn) / len(y), P=pr, R=rc,
                F1=2 * pr * rc / max(1e-9, pr + rc))


def fit_torch(name, Xtr, ytr, Xva, yva, rep):
    torch.manual_seed(2000 + rep)
    m = build(name); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva, dtype=torch.float32); yva_t = torch.tensor(yva)
    best, bad, best_state = -1, 0, None
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        out = m(Xtr_t)
        loss = lossf(out, ytr_t) + 0.01 * sum(p.pow(2).sum() for p in m.parameters())
        loss.backward(); opt.step()
        m.eval()
        f1 = f1_of(yva, m(Xva_t).argmax(1).numpy())["F1"]
        if f1 > best: best, bad, best_state = f1, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PAT: break
    m.load_state_dict(best_state)
    return m


def predict_proba_sk(name, p, X):
    m = SK_BUILD[name](p); m.fit(X["tr"], X["ytr"])
    return m.predict_proba(X["te"])[:, 1]


def main():
    df = pd.read_csv("data/processed.csv"); y = df["y"].values; X = df.drop(columns=["y"]).values
    want = sys.argv[1:] or list(GRID_SK) + TORCH_MODELS
    os.makedirs("out", exist_ok=True)
    preds, probs = [], []
    for rep in range(N_REPS):
        sp = pd.read_csv(f"splits/rep{rep}.csv"); fold = sp["fold"].values
        for k in range(N_FOLDS):
            te = fold == k; tr = ~te
            ytr, yte = y[tr], y[te]
            # 10% validation subset (stratified) for early stopping
            skv = StratifiedKFold(10, shuffle=True, random_state=5000)
            va_idx = next(skv.split(np.zeros(len(ytr)), ytr))[1]
            mask = np.zeros(len(ytr), bool); mask[va_idx] = True
            Xsub, ysub = X[tr][~mask], ytr[~mask]
            Xva, yva = X[tr][mask], ytr[mask]
            for name in want:
                if name in GRID_SK:
                    # inner 3-fold selection on the 90% subset
                    best_p, best_s = None, -1
                    for p in GRID_SK[name]:
                        fs = []
                        for itr, iva in StratifiedKFold(3, shuffle=True, random_state=5000).split(Xsub, ysub):
                            m = SK_BUILD[name](p); m.fit(Xsub[itr], ysub[itr])
                            fs.append(f1_of(ysub[iva], m.predict(Xsub[iva]))["F1"])
                        if np.mean(fs) > best_s: best_s, best_p = np.mean(fs), p
                    pb = predict_proba_sk(name, best_p, dict(tr=Xsub, ytr=ysub, te=X[te]))
                else:
                    m = fit_torch(name, Xsub, ysub, Xva, yva, rep)
                    m.eval()
                    with torch.no_grad():
                        pb = torch.softmax(m(torch.tensor(X[te], dtype=torch.float32)), 1)[:, 1].numpy()
                yp = (pb >= 0.5).astype(int)
                for i, s in enumerate(np.where(te)[0]):
                    preds.append([s, int(yte[i]), int(yp[i]), name, rep, k])
                    probs.append([s, int(yte[i]), float(pb[i]), name, rep, k])
            print(f"rep{rep} fold{k} done", flush=True)
    pd.DataFrame(preds, columns=["sample_id", "y_true", "y_pred", "model", "rep", "fold"]
                 ).to_csv("out/predictions.csv", index=False)
    pd.DataFrame(probs, columns=["sample_id", "y_true", "prob", "model", "rep", "fold"]
                 ).to_csv("out/probabilities.csv", index=False)
    # per-fold metrics -> rebuilds Table S1
    P = pd.DataFrame(preds, columns=["sample_id", "y_true", "y_pred", "model", "rep", "fold"])
    B = pd.DataFrame(probs, columns=["sample_id", "y_true", "prob", "model", "rep", "fold"])
    rows = []
    for (name, rep, k), g in P.groupby(["model", "rep", "fold"]):
        r = f1_of(g.y_true.values, g.y_pred.values); gg = B[(B.model == name) & (B.rep == rep) & (B.fold == k)]
        r["AUC"] = roc_auc_score(gg.y_true, gg.prob); r.update(Model=name, Fold=f"R{rep+1}-F{k+1}")
        rows.append(r)
    pd.DataFrame(rows).to_csv("out/per_fold_metrics.csv", index=False)
    print("saved out/predictions.csv, out/probabilities.csv, out/per_fold_metrics.csv")


if __name__ == "__main__":
    main()
