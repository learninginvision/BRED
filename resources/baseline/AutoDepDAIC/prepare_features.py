"""DAIC-WOZ preprocessing: extract the participant-only audio, then compute and cache mel features.

This runs once. The mel cache (features/mel40) is read by the negative control
code/depaudionet_control.py; the participant-speech extraction functions are
also reused by prepare_w2v2.py.

The cleaning rules and feature parameters follow two open repositories by adbailey1 item by item:
  - https://github.com/adbailey1/daic_woz_process        (preprocessing framework)
  - https://github.com/adbailey1/DepAudioNet_reproduction (DepAudioNet reproduction)
Paper: A. Bailey & M. D. Plumbley, "Gender Bias in Depression Detection
Using Audio Features", EUSIPCO 2021, arXiv:2010.15120

What is taken over:
  1. the four errata tables of config_process.py (the constants below);
  2. the segmentation logic of utils/utilities.py::transcript_file_processing:
     keep only the Participant turns, merge consecutive turns into one interval,
     skip turns marked [sync];
  3. audio/audio_feature_extractor.py::LogMelExtractor: an STFT with a Hann
     window of 1024 and hop 512, squared magnitude, times a mel filter bank
     (40 bins, fmin=0, fmax=sr/2), then SNV normalisation over the whole recording;
  4. the feature hyperparameters of config.py: FEATURE_EXP='mel' (note: power mel
     without the log; create_database() calls mel -> log=False), FREQ_BINS=40,
     WINDOW_SIZE=1024, OVERLAP=50%, SNV=True, SAMPLE_RATE=16000.

Why the virtual interviewer Ellie must be removed: the official documentation
states that Ellie's voice bleeds into the audio, and Burdisso et al.
(arXiv:2404.14463) showed that Ellie's prompts alone reach F1 0.84-0.88. A model
would then learn the prompt pattern rather than the participant's voice, which
is a leak.
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np

DATA_ROOT = os.environ.get("DAIC_ROOT", "/datasets/DAIC-WOZ")
OUT_ROOT = os.path.join(DATA_ROOT, "features", "mel40")

# ---- the four tables below are copied verbatim from daic_woz_process/config_files/config_process.py ----
# intervals (seconds) in which a session was interrupted; they are removed entirely
INTERRUPT = {373: [395, 428], 444: [286, 387]}
# sessions whose transcript has a fixed time offset from the audio; the offset is added to every timestamp
MISALIGNED = {318: 34.319917, 321: 3.8379167, 341: 6.1892, 362: 16.8582}
# error in the official labels: 409 has a PHQ-8 score of 10 (>= 10 means depressed) but a binary label of 0
WRONG_LABELS = {409: 1}
# sessions excluded for technical reasons according to the official documentation
EXCLUDED_SESSIONS = [342, 394, 398, 460]
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000
WINDOW_SIZE = 1024
HOP_SIZE = 512          # OVERLAP = 50%
FREQ_BINS = 40


def participant_spans(transcript_path, session_id):
    """Parse a transcript and return the [start, stop] intervals (seconds) in which the participant speaks.

    Mirrors utilities.py::transcript_file_processing. The original merges
    consecutive Participant turns into one interval (holding_start /
    holding_stop); the same behaviour is kept here.
    """
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
        data = f.readlines()

    offset = MISALIGNED.get(session_id, 0.0)
    interrupt = INTERRUPT.get(session_id)

    # first normalise the lines into a list of turns (start, stop, speaker, is_sync)
    turns = []
    for line in data[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0]) + offset
            stop = float(parts[1]) + offset
        except ValueError:
            continue
        speaker = parts[2]
        is_sync = parts[-1] in ("[sync]", "[syncing]")
        turns.append((start, stop, speaker, is_sync))

    spans = []
    i = 0
    while i < len(turns):
        start, stop, speaker, is_sync = turns[i]
        if speaker != "Participant" or is_sync:
            i += 1
            continue
        # merge the Participant turns that follow immediately
        j = i
        while (j + 1 < len(turns) and turns[j + 1][2] == "Participant"
               and not turns[j + 1][3]):
            j += 1
        merged = (start, turns[j][1])
        i = j + 1

        if interrupt is None:
            spans.append(merged)
            continue
        # clip at the interruption window, same logic as the original
        a, b = merged
        lo, hi = interrupt
        if a < lo < b:
            spans.append((a, lo - 0.01))
        elif a < hi < b:
            spans.append((hi + 0.01, b))
        elif lo <= a <= hi or lo <= b <= hi:
            pass                      # entirely inside the interruption window: dropped
        else:
            spans.append((a, b))
    return spans


def concat_participant_audio(wav_path, spans):
    """Concatenate the participant's speech over the given intervals into one waveform (mirrors modify_audio_file)."""
    audio, sr = librosa.load(wav_path, sr=None)
    if audio.ndim > 1:
        raise ValueError(f"{wav_path} is not mono")
    pieces = []
    for start, stop in spans:
        a, b = int(start * sr), int(stop * sr)
        a, b = max(a, 0), min(b, audio.shape[0])
        if b > a:
            pieces.append(audio[a:b])
    if not pieces:
        return np.zeros(0, dtype=np.float32), sr
    return np.hstack(pieces).astype(np.float32), sr


def snv(x):
    """Standard Normal Variate: subtract the mean and divide by the std over the whole recording (same-named function in the original)."""
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def mel_features(audio, sr):
    """Power mel spectrogram + SNV, returns (40, T). Equivalent to LogMelExtractor(log=False, snv=True)."""
    stft = librosa.stft(y=audio, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                        window=np.hanning(WINDOW_SIZE), center=True,
                        dtype=np.complex64, pad_mode="reflect")
    power = np.abs(stft) ** 2                       # squared=True
    melw = librosa.filters.mel(sr=sr, n_fft=WINDOW_SIZE, n_mels=FREQ_BINS,
                               fmin=0, fmax=sr / 2)
    spec = np.dot(power.T, melw.T)                  # (T, 40)
    spec = spec.astype(np.float32).T                # (40, T)
    return snv(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--force", action="store_true", help="recompute sessions that are already cached")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sessions = sorted(
        int(d.split("_")[0]) for d in os.listdir(DATA_ROOT) if d.endswith("_P")
    )
    manifest = {}
    skipped = []
    for k, sid in enumerate(sessions, 1):
        if sid in EXCLUDED_SESSIONS:
            skipped.append((sid, "officially excluded"))
            continue
        base = os.path.join(DATA_ROOT, f"{sid}_P")
        wav = os.path.join(base, f"{sid}_AUDIO.wav")
        txt = os.path.join(base, f"{sid}_TRANSCRIPT.csv")
        if not (os.path.exists(wav) and os.path.exists(txt)):
            skipped.append((sid, "files missing (the download may be incomplete)"))
            continue

        out_path = os.path.join(args.out, f"{sid}.npy")
        if os.path.exists(out_path) and not args.force:
            feat = np.load(out_path, mmap_mode="r")
            manifest[sid] = {"frames": int(feat.shape[1])}
            continue

        spans = participant_spans(txt, sid)
        audio, sr = concat_participant_audio(wav, spans)
        if audio.shape[0] < WINDOW_SIZE:
            skipped.append((sid, f"participant speech too short ({audio.shape[0]} samples)"))
            continue
        if sr != SAMPLE_RATE:
            print(f"  warning {sid}: sampling rate {sr} != {SAMPLE_RATE}")
        feat = mel_features(audio, sr)
        np.save(out_path, feat)
        manifest[sid] = {
            "frames": int(feat.shape[1]),
            "turns": len(spans),
            "seconds": round(audio.shape[0] / sr, 1),
        }
        print(f"[{k}/{len(sessions)}] {sid}: {len(spans)} turns "
              f"{audio.shape[0]/sr:.0f}s -> mel {feat.shape}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({str(k): v for k, v in sorted(manifest.items())}, f, indent=2)
    print(f"\ndone: {len(manifest)} sessions, {len(skipped)} skipped")
    for sid, why in skipped[:20]:
        print(f"  skipped {sid}: {why}")
    if manifest:
        fr = [v["frames"] for v in manifest.values()]
        print(f"frames min={min(fr)} max={max(fr)} median={int(np.median(fr))}")


if __name__ == "__main__":
    sys.exit(main())
