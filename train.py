"""Train AnemiaScan: color features -> StandardScaler -> MLP (ANN). Run: python train.py <dataset_folder>"""
import os, re, sys, hashlib, json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from features import extract_features

folder = sys.argv[1]
rows, seen = [], set()
for f in sorted(os.listdir(folder)):
    if not f.lower().endswith(".png"): continue
    p = os.path.join(folder, f)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in seen: continue                      # drop exact duplicates
    seen.add(h)
    g = re.sub(r" - copy.*| ?\(\d+\)", "", f.lower().replace(".png", ""))
    g = re.sub(r"(fv|ra)$", "", g)              # augmented variants share one group
    rows.append((extract_features(Image.open(p)), 0 if f.lower().startswith("non") else 1, g))
X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); groups = np.array([r[2] for r in rows])
print(f"{len(y)} unique images | anemic={y.sum()} non-anemic={(y==0).sum()} | source groups={len(set(groups))}")

# Group split: copies of the same eyelid never land in both train and test (avoids fake-high accuracy)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))

# Overfitting control: small network + strong L2 regularization (alpha) + limited training iterations.
# (Chosen by comparing train-test gaps over 8 group splits; bigger/weaker-regularized nets memorize the training eyelids.)
best = make_pipeline(StandardScaler(),
                     MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=300, random_state=42))
best.fit(X[tr], y[tr])
print("Train accuracy:", round(accuracy_score(y[tr], best.predict(X[tr])), 3),
      "| Test accuracy:", round(accuracy_score(y[te], best.predict(X[te])), 3))

rf = RandomForestClassifier(300, random_state=42).fit(X[tr], y[tr])   # baseline for comparison
report = {}
for name, m in [("MLP (ANN)", best), ("RandomForest (baseline)", rf)]:
    p, pr = m.predict(X[te]), m.predict_proba(X[te])[:, 1]
    report[name] = dict(accuracy=accuracy_score(y[te], p), sensitivity=recall_score(y[te], p),
                        precision=precision_score(y[te], p), auc=roc_auc_score(y[te], pr))
    print(f"\n{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in report[name].items()))
print("\n", classification_report(y[te], best.predict(X[te]), target_names=["Non-anemic", "Anemic"]))

ConfusionMatrixDisplay(confusion_matrix(y[te], best.predict(X[te])), display_labels=["Non-anemic", "Anemic"]).plot(cmap="Reds")
plt.title("MLP - test set"); plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
joblib.dump(best, "anemia_model.joblib")
json.dump(report, open("metrics.json", "w"), indent=2)
print("Saved anemia_model.joblib, metrics.json, confusion_matrix.png")
