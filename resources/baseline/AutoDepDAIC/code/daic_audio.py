"""Participant-speech extraction for DAIC-WOZ (a self-contained copy inside code/).

The functions are the same as those of the same name in ../prepare_features.py.
They are copied into code/ because an experiment workspace carries only the
code/ directory, and the experiment must be able to rebuild the features from
the audio when the cache is missing. The cleaning rules come from
github.com/adbailey1/daic_woz_process (the port was verified against the
original repository). If you change the extraction logic, change
../prepare_features.py as well.

Why only the participant's turns are used: Burdisso et al. (arXiv:2404.14463)
showed that the prompts of the virtual interviewer Ellie alone reach F1
0.84-0.88. That is a leak of the interview protocol, not a signal in the voice.
"""
import librosa
import numpy as np

# ---- errata tables copied verbatim from daic_woz_process/config_files/config_process.py ----
INTERRUPT = {373: [395, 428], 444: [286, 387]}
MISALIGNED = {318: 34.319917, 321: 3.8379167, 341: 6.1892, 362: 16.8582}
WRONG_LABELS = {409: 1}   # 409 has PHQ-8 = 10 (depressed), but the official binary label is 0
SAMPLE_RATE = 16000


def participant_spans(transcript_path, session_id):
    """Parse a transcript and return the [start, stop] intervals (seconds) in which the participant speaks.

    Consecutive Participant turns are merged into one interval, [sync] turns are
    skipped, and the time-offset errata and the interruption windows are applied.
    """
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
        data = f.readlines()

    offset = MISALIGNED.get(session_id, 0.0)
    interrupt = INTERRUPT.get(session_id)

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
        j = i
        while (j + 1 < len(turns) and turns[j + 1][2] == "Participant"
               and not turns[j + 1][3]):
            j += 1
        merged = (start, turns[j][1])
        i = j + 1

        if interrupt is None:
            spans.append(merged)
            continue
        a, b = merged
        lo, hi = interrupt
        if a < lo < b:
            spans.append((a, lo - 0.01))
        elif a < hi < b:
            spans.append((hi + 0.01, b))
        elif lo <= a <= hi or lo <= b <= hi:
            pass
        else:
            spans.append((a, b))
    return spans


def concat_participant_audio(wav_path, spans):
    """Concatenate the participant's speech over the given intervals into one waveform."""
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
