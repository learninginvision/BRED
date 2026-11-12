"""DAIC-WOZ speech depression detection baseline: frozen wav2vec 2.0 features + a linear probe.

Source of the method (a published recipe built from open components, not our own):
  - Frozen self-supervised speech representations with a light classification
    head are a standard recipe in the literature:
    Wu, Zhang & Woodland, "Self-supervised representations in speech-based
    depression detection", ICASSP 2023 (layer-wise evaluation of frozen SSL features);
    Zhang et al., Scientific Reports 14:9543 (2024) (wav2vec2 transfer learning).
  - All components are open: Hugging Face wav2vec2-base + scikit-learn.

Evaluation protocol (taken from HAREN-CTC, arXiv:2510.08593v2, Sec. 4.5,
"generalization evaluation"):
  - pool all usable subjects of train + dev + test (188 subjects, 56 depressed;
    the source files of subject 440 are corrupt)
  - subject-level stratified 5-fold CV x 5 random seeds
  - the decision threshold is tuned only on out-of-fold (OOF) predictions inside
    the training folds (maximising macro F1); the evaluated fold takes no part
    in training, tuning or selection
  - metrics are averaged over folds, then over seeds; the primary metric is
    macro_f1, with auc as the secondary one

Three hard rules (failure modes with published evidence on this corpus):
  1. Use only the participant's own speech (daic_audio.py cuts it out by the
     transcript timestamps). Anything about the virtual interviewer Ellie
     (turns, durations, gaps) is a protocol leak.
  2. Select nothing on the evaluated fold. The convention in the literature
     (select on dev, then report dev) inflates scores systematically by ~28%
     (HAREN-CTC itself reports .81 -> .576).
  3. To judge whether a gain is real, look at auc and the seed-to-seed std as
     well: a macro_f1 gain below 0.02 counts as noise.

Reference points under the same protocol:
  - this baseline (layer 12, mean, C=0.1): macro_f1 .561±.025 / auc .631±.024
  - DepAudioNet negative control (depaudionet_control.py): .474±.033 / .467±.045 (chance level)
  - literature SOTA HAREN-CTC (code not released): .576±.02 / .599±.02
  - the degenerate all-positive solution scores macro_f1 ≈ .23 on the 188-subject pool

Feature cache: {DAIC_ROOT}/features/w2v2_layers/{sid}.npy, shape (13, n_seg, 768).
All 13 layers are cached, so switching to another layer or fusing several needs
no re-extraction (set LAYER). If the cache is missing, this script rebuilds it
from the audio (about 4 minutes for 188 subjects on an RTX 2080 Ti; this happens once).
"""
import argparse
import json
import os
import time

# Limit the BLAS/OpenMP threads so that experiments running in parallel do not
# oversubscribe the CPU (without the limit, a 14 s run burned 6.5 CPU-minutes on
# contention; 8 threads are enough for the matrix sizes of this task)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from daic_audio import WRONG_LABELS, concat_participant_audio, participant_spans

# -------------- configuration (the part experiments mostly change) --------------
DATA_ROOT = os.environ.get("DAIC_ROOT", "/datasets/DAIC-WOZ")
CACHE_DIR = os.path.join(DATA_ROOT, "features", "w2v2_layers")
MODEL_PATH = os.environ.get("W2V2_PATH", "/datasets/wav2vec2-base")   # frozen SSL encoder
SEG_SECONDS = 5           # length of the participant-speech segments (seconds)
LAYER = 12                # which hidden layer (0 = CNN output, 1..12 = transformer layers)
SUBJECT_POOL = "mean"     # segment vectors -> subject vector: mean | meanstd
C = 0.1                   # inverse regularisation strength of the logistic regression
N_FOLDS = 5
SEEDS = (0, 1, 2, 3, 4)
# --------------------------------------------------------------------------


def load_split(csv_name, label_col="PHQ8_Binary"):
    df = pd.read_csv(os.path.join(DATA_ROOT, csv_name))
    ids, labels = [], []
    for _, row in df.iterrows():
        sid = int(row["Participant_ID"])
        if not os.path.exists(os.path.join(DATA_ROOT, f"{sid}_P",
                                           f"{sid}_TRANSCRIPT.csv")):
            continue                      # subject 440: source files corrupt, no transcript
        ids.append(sid)
        labels.append(WRONG_LABELS.get(sid, int(row[label_col])))
    return np.array(ids), np.array(labels, dtype=int)


def load_pool():
    tr = load_split("train_split_Depression_AVEC2017.csv")
    dv = load_split("dev_split_Depression_AVEC2017.csv")
    te = load_split("full_test_split.csv", label_col="PHQ_Binary")
    return np.concatenate([tr[0], dv[0], te[0]]), \
        np.concatenate([tr[1], dv[1], te[1]])


def ensure_cache(ids):
    """Extract the features of subjects missing from the cache (cold start; cached from then on)."""
    todo = [s for s in ids if not os.path.exists(f"{CACHE_DIR}/{s}.npy")]
    if not todo:
        return
    print(f"cache missing for {len(todo)} subjects; extracting now (about {max(1, round(len(todo)*1.5/60))} "
          f"min)...", flush=True)
    import torch
    from transformers import Wav2Vec2Model
    os.makedirs(CACHE_DIR, exist_ok=True)
    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Model.from_pretrained(MODEL_PATH).to(dev_t).eval()
    seg_len = SEG_SECONDS * 16000
    for k, sid in enumerate(todo, 1):
        spans = participant_spans(
            f"{DATA_ROOT}/{sid}_P/{sid}_TRANSCRIPT.csv", sid)
        audio, _ = concat_participant_audio(
            f"{DATA_ROOT}/{sid}_P/{sid}_AUDIO.wav", spans)
        if audio.shape[0] < seg_len:
            audio = np.pad(audio, (0, seg_len - audio.shape[0]))
        n = audio.shape[0] // seg_len
        segs = audio[:n * seg_len].reshape(n, seg_len)
        segs = (segs - segs.mean(1, keepdims=True)) / \
            (segs.std(1, keepdims=True) + 1e-7)
        outs = []
        with torch.no_grad():
            for i in range(0, n, 32):
                xb = torch.from_numpy(segs[i:i + 32]).float().to(dev_t)
                hs = model(xb, output_hidden_states=True).hidden_states
                outs.append(torch.stack(hs).mean(2).cpu().numpy())
        np.save(f"{CACHE_DIR}/{sid}.npy",
                np.concatenate(outs, axis=1).astype(np.float32))
        if k % 10 == 0:
            print(f"  ...{k}/{len(todo)}", flush=True)


def subject_matrix(ids):
    """Subject-level matrix: read the segment embeddings of the chosen layer and pool them per subject."""
    rows = []
    for sid in ids:
        E = np.load(f"{CACHE_DIR}/{sid}.npy", mmap_mode="r")
        L = np.asarray(E[LAYER])                       # (n_seg, 768)
        rows.append(np.concatenate([L.mean(0), L.std(0)])
                    if SUBJECT_POOL == "meanstd" else L.mean(0))
    return np.stack(rows)


def fold_metrics(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    return dict(
        macro_f1=f1_score(y_true, pred, average="macro", zero_division=0),
        uar=recall_score(y_true, pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, pred, average="macro",
                                        zero_division=0),
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, pred, pos_label=1, zero_division=0),
        f1_nondep=f1_score(y_true, pred, pos_label=0, zero_division=0),
        precision=precision_score(y_true, pred, pos_label=1, zero_division=0),
        recall=recall_score(y_true, pred, pos_label=1, zero_division=0),
        accuracy=accuracy_score(y_true, pred),
    )


def tune_threshold(oof, y):
    """Threshold with the best macro F1 on the OOF predictions of the training folds (the evaluated fold is never used)."""
    cand = np.unique(np.round(oof, 3))
    return max(cand, key=lambda c: f1_score(y, (oof >= c).astype(int),
                                            average="macro", zero_division=0))


def main():
    global LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "metrics.json"))
    args = ap.parse_args()
    LAYER = args.layer

    ids, ys = load_pool()
    print(f"pooled set: {len(ids)} subjects ({ys.sum()} depressed) | layer={LAYER} "
          f"pool={SUBJECT_POOL} C={C}", flush=True)
    ensure_cache(ids)
    X = subject_matrix(ids)

    t0 = time.time()
    mk = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, C=C,
                                          class_weight="balanced"))
    seed_means = []
    for seed in args.seeds:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        fms = []
        for tr_idx, va_idx in skf.split(X, ys):
            Xtr, ytr = X[tr_idx], ys[tr_idx]
            oof = cross_val_predict(
                clone(mk), Xtr, ytr,
                cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                method="predict_proba")[:, 1]
            thr = tune_threshold(oof, ytr)
            model = clone(mk).fit(Xtr, ytr)
            prob = model.predict_proba(X[va_idx])[:, 1]
            fms.append(fold_metrics(ys[va_idx], prob, thr))
        seed_means.append({k: float(np.nanmean([f[k] for f in fms]))
                           for k in fms[0]})
        print(f"  seed={seed} macro_f1={seed_means[-1]['macro_f1']:.4f} "
              f"auc={seed_means[-1]['auc']:.4f}", flush=True)

    metrics = {}
    for k in seed_means[0]:
        vals = [s[k] for s in seed_means]
        metrics[k] = float(np.mean(vals))
        metrics[f"{k}_std"] = float(np.std(vals))
    metrics.update(n_subjects=int(len(ids)), n_depressed=int(ys.sum()),
                   n_seeds=len(args.seeds), n_folds=args.folds,
                   protocol="merged-pool subject-stratified CV",
                   layer=LAYER, subject_pool=SUBJECT_POOL, C=C)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 72)
    print(f"frozen w2v2 probe | layer={LAYER} | {len(args.seeds)} seeds x "
          f"{args.folds} folds | {time.time()-t0:.0f}s")
    print(f"  macro_f1 = {metrics['macro_f1']:.4f} ± {metrics['macro_f1_std']:.4f}"
          f"  (primary metric)")
    print(f"  auc      = {metrics['auc']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"  uar={metrics['uar']:.4f}  macro_p={metrics['macro_precision']:.4f}"
          f"  F1(dep)={metrics['f1']:.4f}  F1(non-dep)={metrics['f1_nondep']:.4f}")
    print("  ref: DepAudioNet negative control (same protocol)  macro_f1 .474 / auc .467 (chance level)")
    print("  ref: HAREN-CTC, literature SOTA (not released)     macro_f1 .576 / auc .599")
    print("  ref: degenerate all-positive solution              macro_f1 ≈ .23")
    print("=" * 72)


if __name__ == "__main__":
    main()
