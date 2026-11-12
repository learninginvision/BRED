"""DepAudioNet: the NEGATIVE CONTROL of this task, not the baseline to iterate on
(the baseline is experiment.py).

Role: under the leakage-free protocol this model scored at chance level in all
six evaluations (pooled 188-subject CV: macro_f1 .474±.033 / auc .467±.045).
It is kept as a negative control of the evaluation protocol: a model without
signal scores at chance, which shows that the protocol gives nothing away, so
the .561/.631 of the probe in experiment.py is a real signal. This file does
not need to be run again; the control values only have to be re-measured if the
evaluation protocol itself changes.

The original notes follow.

Source of the method (a published method with an open implementation, not our own):
  - X. Ma, H. Yang, Q. Chen, D. Huang, Y. Wang, "DepAudioNet: An Efficient Deep
    Model for Audio based Depression Classification", AVEC'16 workshop @ ACM MM.
    The standard audio baseline on DAIC-WOZ; most later papers compare against it.
  - Open reproduction: https://github.com/adbailey1/DepAudioNet_reproduction
    (A. Bailey & M. D. Plumbley, EUSIPCO 2021, arXiv:2010.15120)
    Validation results reported in that repository's README:
    F1 (non-depressed) .732 / F1 (depressed) .522 / mean F1 .627

The architecture (CustomMel7) and the hyperparameters follow that repository item by item:
  Conv1d(40->128, k=3, s=1, p=1) + BatchNorm + ReLU
  MaxPool1d(k=3, s=3) -> Dropout(0.05)
  LSTM(128 -> 128, 2 layers, unidirectional) -> last frame -> Linear(128->1) + Sigmoid
  Adam lr=1e-3, learning rate x0.9 every 2 epochs, batch=20, 100 epochs, 5 random seeds

The data protocol is taken over as well:
  - Features: power mel spectrogram of the participant's speech (40 bins, window
    1024, hop 512, SNV over the whole recording), cached beforehand by
    ../prepare_features.py
  - Training set: each subject is randomly cropped to the length of the shortest
    training recording and cut into 120-frame segments; every epoch the
    non-depressed subjects are randomly downsampled so that both classes have
    the same number of subjects
  - Validation and test sets are not cropped; all segments of the full recording are used
  - Training is per segment, evaluation per subject: the sigmoid outputs of all
    segments of a subject are averaged, and >= 0.5 counts as depressed

Evaluation protocol (the only substantive difference from the original repository):
  The original main1.py:548-564 picks the epoch with a criterion that includes the
  dev F1 and then reports the dev result as the final score, i.e. it selects the
  model on the reported set. By default this file instead splits an inner
  validation set off the training set to pick the epoch; dev takes no part in any
  selection and is evaluated once at the end. --select dev switches back to the
  original convention for comparison. The test set (47 subjects, labels in
  full_test_split.csv) is not touched by default; it is evaluated only with --final.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = os.environ.get("DAIC_ROOT", "/datasets/DAIC-WOZ")
FEAT_DIR = os.path.join(DATA_ROOT, "features", "mel40")

FREQ_BINS = 40
SEGMENT_DIM = 120        # FEATURE_DIMENSIONS
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
LR_DECAY_EVERY = 2       # learn_rate_factor
LR_DECAY = 0.9
TOTAL_EPOCHS = 100
SEEDS = (0, 1, 2, 3, 4)  # EXP_RUNTHROUGH = 5
WRONG_LABELS = {409: 1}  # error in the official labels, see prepare_features.py

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- data -----------------------------
def load_split(csv_name, id_col="Participant_ID", label_col="PHQ8_Binary"):
    df = pd.read_csv(os.path.join(DATA_ROOT, csv_name))
    ids, labels = [], []
    for _, row in df.iterrows():
        sid = int(row[id_col])
        path = os.path.join(FEAT_DIR, f"{sid}.npy")
        if not os.path.exists(path):
            continue
        ids.append(sid)
        labels.append(WRONG_LABELS.get(sid, int(row[label_col])))
    return np.array(ids), np.array(labels, dtype=np.float32)


def load_features(ids):
    return {sid: np.load(os.path.join(FEAT_DIR, f"{sid}.npy")) for sid in ids}


def segment(feat, seg_dim=SEGMENT_DIM):
    """Cut (40, T) into (n, 40, seg_dim), zero-padding the last segment. Mirrors feature_segmenter."""
    t = feat.shape[1]
    n = t // seg_dim + (1 if t % seg_dim else 0)
    pad = n * seg_dim - t
    if pad:
        feat = np.hstack((feat, np.zeros((FREQ_BINS, pad), dtype=feat.dtype)))
    return np.stack(np.split(feat, n, axis=1))


def build_train_batchset(feats, ids, labels, min_len, rng):
    """Training set of one epoch: class downsampling + random crop + segmentation."""
    dep = ids[labels == 1]
    ndep = ids[labels == 0]
    keep_ndep = rng.choice(ndep, size=min(len(dep), len(ndep)), replace=False)
    chosen = np.concatenate([dep, keep_ndep])

    X, y = [], []
    lab = dict(zip(ids, labels))
    for sid in chosen:
        f = feats[sid]
        room = f.shape[1] - min_len
        if room > 0:
            start = rng.integers(0, room + 1)
            f = f[:, start:start + min_len]
        segs = segment(f)
        X.append(segs)
        y.append(np.full(len(segs), lab[sid], dtype=np.float32))
    return np.concatenate(X), np.concatenate(y)


def build_eval_set(feats, ids):
    """Evaluation set: no cropping, the full recording is segmented, and the subject of each segment is recorded."""
    X, owner = [], []
    for sid in ids:
        segs = segment(feats[sid])
        X.append(segs)
        owner.append(np.full(len(segs), sid))
    return np.concatenate(X), np.concatenate(owner)


# ----------------------------- model -----------------------------
class DepAudioNet(nn.Module):
    """CustomMel7, layer by layer as in DepAudioNet_reproduction/exp_run/models_pytorch.py."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(FREQ_BINS, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):                      # x: (B, 40, 120)
        x = self.relu(self.bn(self.conv(x)))
        x = self.drop(self.pool(x))
        x = x.transpose(1, 2)                  # (B, T, 128)
        x, _ = self.lstm(x)
        return torch.sigmoid(self.fc(x[:, -1, :])).squeeze(-1)


# ------------------------ training / evaluation ------------------------
@torch.no_grad()
def subject_scores(model, X, owner, ids):
    """Average the segment probabilities per subject -> subject-level scores."""
    model.eval()
    probs = []
    for i in range(0, len(X), 256):
        xb = torch.from_numpy(X[i:i + 256]).float().to(dev)
        probs.append(model(xb).cpu().numpy())
    probs = np.concatenate(probs)
    return np.array([probs[owner == sid].mean() for sid in ids])


def score_metrics(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    out = dict(
        f1=f1_score(y_true, pred, pos_label=1, zero_division=0),
        macro_f1=f1_score(y_true, pred, average="macro", zero_division=0),
        f1_nondep=f1_score(y_true, pred, pos_label=0, zero_division=0),
        precision=precision_score(y_true, pred, pos_label=1, zero_division=0),
        recall=recall_score(y_true, pred, pos_label=1, zero_division=0),
        accuracy=accuracy_score(y_true, pred),
    )
    # Primary metric: the mean of the depressed-class and non-depressed-class F1
    # (= binary macro-F1, the "F1-Avg" of the literature). The depressed-class F1
    # alone is not used because the trivial "all depressed" solution already scores
    # 0.511 on dev, while its f1_avg is only 0.255; only the latter penalises it.
    out["f1_avg"] = (out["f1"] + out["f1_nondep"]) / 2
    out["auc"] = (roc_auc_score(y_true, y_prob)
                  if len(set(y_true.tolist())) > 1 else float("nan"))
    return out


def run_seed(seed, feats, tr_ids, tr_y, sel_ids, sel_y, eval_sets, epochs,
             verbose=True):
    """Train one seed. sel_* is the selection set used to pick the epoch; eval_sets are the final evaluation sets."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    min_len = min(feats[s].shape[1] for s in tr_ids)
    model = DepAudioNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.BCELoss()

    Xsel, osel = build_eval_set(feats, sel_ids)
    best = (-1.0, -1)                       # (f1_avg on the selection set, epoch)
    best_state = None

    for ep in range(1, epochs + 1):
        X, y = build_train_batchset(feats, tr_ids, tr_y, min_len, rng)
        perm = rng.permutation(len(X))
        model.train()
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            if len(b) < 2:                  # BatchNorm needs more than one sample
                continue
            xb = torch.from_numpy(X[b]).float().to(dev)
            yb = torch.from_numpy(y[b]).float().to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        if ep % LR_DECAY_EVERY == 0:
            for g in opt.param_groups:
                g["lr"] *= LR_DECAY

        # The selection criterion must match the reported primary metric. Picking the
        # epoch by the depressed-class F1, as an earlier version did, was a mistake:
        # it favours the high-recall operating point that labels most subjects as
        # depressed, giving recall .73 / precision .41 and a non-depressed F1 of only
        # .51 (.73 in the literature).
        sel = score_metrics(sel_y, subject_scores(model, Xsel, osel, sel_ids))["f1_avg"]
        if sel > best[0]:
            best = (sel, ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 20 == 0:
            print(f"    seed={seed} epoch={ep:3d} selection f1_avg={sel:.4f} "
                  f"(best {best[0]:.4f} @ep{best[1]})", flush=True)

    model.load_state_dict(best_state)
    results = {}
    for name, (ids, y) in eval_sets.items():
        Xe, oe = build_eval_set(feats, ids)
        results[name] = score_metrics(y, subject_scores(model, Xe, oe, ids))
    results["_selected_epoch"] = best[1]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["inner", "dev"], default="inner",
                    help="set used to pick the epoch: inner = a split inside the training set (clean); "
                         "dev = dev itself (the convention of the original repository and the literature)")
    ap.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--final", action="store_true",
                    help="also evaluate once on the 47-subject test set (keep it off normally)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "metrics.json"))
    args = ap.parse_args()

    tr_ids, tr_y = load_split("train_split_Depression_AVEC2017.csv")
    dv_ids, dv_y = load_split("dev_split_Depression_AVEC2017.csv")
    te_ids, te_y = load_split("full_test_split.csv", label_col="PHQ_Binary")
    print(f"train {len(tr_ids)} subjects ({int(tr_y.sum())} depressed) | "
          f"dev {len(dv_ids)} subjects ({int(dv_y.sum())} depressed) | "
          f"test {len(te_ids)} subjects ({int(te_y.sum())} depressed)", flush=True)

    need = list(tr_ids) + list(dv_ids) + (list(te_ids) if args.final else [])
    feats = load_features(need)

    eval_sets = {"dev": (dv_ids, dv_y)}
    if args.final:
        eval_sets["test"] = (te_ids, te_y)

    t0 = time.time()
    runs = []
    for seed in args.seeds:
        if args.select == "dev":
            sub_tr_ids, sub_tr_y = tr_ids, tr_y
            sel_ids, sel_y = dv_ids, dv_y
        else:
            # a stratified 20% of the training set is the inner validation set; dev takes no part in selection
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            fit_idx, val_idx = next(iter(skf.split(tr_ids, tr_y)))
            sub_tr_ids, sub_tr_y = tr_ids[fit_idx], tr_y[fit_idx]
            sel_ids, sel_y = tr_ids[val_idx], tr_y[val_idx]
        r = run_seed(seed, feats, sub_tr_ids, sub_tr_y, sel_ids, sel_y,
                     eval_sets, args.epochs)
        runs.append(r)
        print(f"  seed={seed} dev f1_avg={r['dev']['f1_avg']:.4f} "
              f"F1(dep)={r['dev']['f1']:.4f} (ep{r['_selected_epoch']})", flush=True)

    metrics = {}
    for split in eval_sets:
        for k in runs[0][split]:
            vals = [r[split][k] for r in runs]
            prefix = "" if split == "dev" else f"{split}_"
            metrics[f"{prefix}{k}"] = float(np.nanmean(vals))
            metrics[f"{prefix}{k}_std"] = float(np.nanstd(vals))
    metrics["select_mode"] = args.select
    metrics["n_seeds"] = len(runs)

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 72)
    print(f"DepAudioNet | selection={args.select} | {len(runs)} seeds | "
          f"{time.time()-t0:.0f}s")
    print(f"  dev  f1_avg={metrics['f1_avg']:.4f} ± {metrics['f1_avg_std']:.4f}  "
          f"(primary metric)")
    print(f"       F1(dep)={metrics['f1']:.4f} ± {metrics['f1_std']:.4f}  "
          f"F1(non-dep)={metrics['f1_nondep']:.4f}")
    print(f"       P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f}")
    if args.final:
        print(f"  test f1_avg={metrics['test_f1_avg']:.4f} ± "
              f"{metrics['test_f1_avg_std']:.4f}  "
              f"F1(dep)={metrics['test_f1']:.4f}  AUC={metrics['test_auc']:.4f}")
    print("  literature ref, DepAudioNet (adbailey1 reproduction): F1(dep) .522 / f1_avg .627")
    print("  trivial ref, everyone predicted depressed           : F1(dep) .511 / f1_avg .255")
    print("=" * 72)


if __name__ == "__main__":
    main()
