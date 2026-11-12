"""Run the participant's speech through frozen wav2vec2-base and cache the
segment-level time averages of all 13 hidden layers.

Cache: {DAIC_ROOT}/features/w2v2_layers/{sid}.npy, shape (13, n_seg, 768), float32.
  - 13 = the embedding layer (CNN output) + 12 transformer layers (indices 0..12, 12 is the last)
  - n_seg = the number of 5 s segments cut from the subject's speech
  - the hidden states of each segment are already averaged over time

Why extract once and cache every layer: layer selection and multi-layer fusion
are the largest single lever in the literature (the HAREN-CTC ablation shows
macro F1 ranging over .38-.70 between single layers), and with all layers cached
such experiments need no re-extraction.

Segmentation and normalisation must match code/experiment.py:
participant speech (cut out by the transcript timestamps and concatenated) ->
non-overlapping 5 s segments -> SNV normalisation per segment.

Usage: CUDA_VISIBLE_DEVICES=0 python prepare_w2v2.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_features as P  # noqa: E402  reuse the verified participant-speech extraction

R = P.DATA_ROOT
CACHE = os.path.join(R, "features", "w2v2_layers")
MODEL = os.environ.get("W2V2_PATH", "/datasets/wav2vec2-base")
SEG = 5 * 16000
BATCH = 32
dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_sids():
    sids = set()
    for csv in ("train_split_Depression_AVEC2017.csv",
                "dev_split_Depression_AVEC2017.csv",
                "full_test_split.csv"):
        sids |= set(int(s) for s in pd.read_csv(os.path.join(R, csv)).Participant_ID)
    # subject 440: the source files are corrupt and there is no transcript, so the speech cannot be extracted
    return [s for s in sorted(sids)
            if os.path.exists(f"{R}/{s}_P/{s}_TRANSCRIPT.csv")]


def extract(sid, model):
    spans = P.participant_spans(f"{R}/{sid}_P/{sid}_TRANSCRIPT.csv", sid)
    audio, _ = P.concat_participant_audio(f"{R}/{sid}_P/{sid}_AUDIO.wav", spans)
    if audio.shape[0] < SEG:
        audio = np.pad(audio, (0, SEG - audio.shape[0]))
    n = audio.shape[0] // SEG
    segs = audio[:n * SEG].reshape(n, SEG)
    segs = (segs - segs.mean(1, keepdims=True)) / (segs.std(1, keepdims=True) + 1e-7)
    outs = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            xb = torch.from_numpy(segs[i:i + BATCH]).float().to(dev_t)
            hs = model(xb, output_hidden_states=True).hidden_states  # 13 x (B,T,768)
            outs.append(torch.stack(hs).mean(2).cpu().numpy())       # (13,B,768)
    return np.concatenate(outs, axis=1)                              # (13,n_seg,768)


def main():
    os.makedirs(CACHE, exist_ok=True)
    sids = all_sids()
    todo = [s for s in sids if not os.path.exists(f"{CACHE}/{s}.npy")]
    print(f"{len(sids)} subjects in total, {len(todo)} to extract", flush=True)
    if not todo:
        return
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(MODEL).to(dev_t).eval()
    t0 = time.time()
    for k, sid in enumerate(todo, 1):
        E = extract(sid, model)
        np.save(f"{CACHE}/{sid}.npy", E.astype(np.float32))
        if k % 10 == 0 or k == len(todo):
            eta = (time.time() - t0) / k * (len(todo) - k)
            print(f"  {k}/{len(todo)} (latest {sid}: {E.shape}) "
                  f"elapsed {(time.time()-t0)/60:.1f}min, about {eta/60:.1f}min left",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
