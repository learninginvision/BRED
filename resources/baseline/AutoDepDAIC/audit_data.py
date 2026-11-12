"""Integrity check of the DAIC-WOZ data.

Checks the local copy item by item against the official documentation
(DAICWOZDepression_Documentation_AVEC2017.pdf), to confirm that it is complete
and consistent with the literature before the baseline's result on it is used
as the floor.
"""
import os
import wave

import numpy as np
import pandas as pd

ROOT = os.environ.get("DAIC_ROOT", "/datasets/DAIC-WOZ")
EXCLUDED = [342, 394, 398, 460]
MISALIGNED = {318, 321, 341, 362}
INTERRUPT = {373, 444}
NO_ELLIE = {451, 458, 480}


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        mark = "OK  " if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{mark}] {label}{('  -> ' + detail) if detail else ''}")

    print("=" * 74)
    print("1. Number of sessions and their ids")
    dirs = sorted(int(d.split("_")[0]) for d in os.listdir(ROOT) if d.endswith("_P"))
    check(f"{len(dirs)} session directories (official: 189)", len(dirs) == 189)
    check("ids range from 300 to 492", dirs and dirs[0] == 300 and dirs[-1] == 492,
          f"found {dirs[0]}-{dirs[-1]}" if dirs else "none")
    expected = [i for i in range(300, 493) if i not in EXCLUDED]
    missing = sorted(set(expected) - set(dirs))
    extra = sorted(set(dirs) - set(expected))
    check("exactly 300-492 without 342/394/398/460",
          not missing and not extra,
          f"missing {missing[:10]}{'...' if len(missing) > 10 else ''} extra {extra}")

    print("\n2. Split files and labels")
    tr = pd.read_csv(f"{ROOT}/train_split_Depression_AVEC2017.csv")
    dv = pd.read_csv(f"{ROOT}/dev_split_Depression_AVEC2017.csv")
    te = pd.read_csv(f"{ROOT}/full_test_split.csv")
    check(f"train: {len(tr)} subjects / {int(tr.PHQ8_Binary.sum())} depressed (official: 107/30)",
          len(tr) == 107 and tr.PHQ8_Binary.sum() == 30)
    check(f"dev: {len(dv)} subjects / {int(dv.PHQ8_Binary.sum())} depressed (official: 35/12)",
          len(dv) == 35 and dv.PHQ8_Binary.sum() == 12)
    check(f"test: {len(te)} subjects / {int(te.PHQ_Binary.sum())} depressed (official: 47/14)",
          len(te) == 47 and te.PHQ_Binary.sum() == 14)
    allids = set(tr.Participant_ID) | set(dv.Participant_ID) | set(te.Participant_ID)
    check(f"the three splits cover {len(allids)} subjects without overlap",
          len(allids) == 189 and len(tr) + len(dv) + len(te) == 189)
    check("the splits cover every session directory", allids == set(dirs),
          f"difference {sorted(allids ^ set(dirs))[:10]}")

    print("\n3. Consistency of the PHQ-8 labels (Binary should equal Score >= 10)")
    for name, df, b, s in [("train", tr, "PHQ8_Binary", "PHQ8_Score"),
                           ("dev", dv, "PHQ8_Binary", "PHQ8_Score"),
                           ("test", te, "PHQ_Binary", "PHQ_Score")]:
        bad = df[df[b] != (df[s] >= 10).astype(int)]
        ids = list(bad.Participant_ID)
        if name == "train":
            check(f"{name}: the only inconsistency is the known label error of 409",
                  ids == [409], f"found {ids}")
        else:
            check(f"{name}: no inconsistency", len(bad) == 0, f"found {ids}")

    print("\n4. Required files of each session")
    miss_wav, miss_txt, miss_cov, miss_for = [], [], [], []
    for sid in dirs:
        d = f"{ROOT}/{sid}_P"
        if not os.path.exists(f"{d}/{sid}_AUDIO.wav"):
            miss_wav.append(sid)
        if not os.path.exists(f"{d}/{sid}_TRANSCRIPT.csv"):
            miss_txt.append(sid)
        if not os.path.exists(f"{d}/{sid}_COVAREP.csv"):
            miss_cov.append(sid)
        if not os.path.exists(f"{d}/{sid}_FORMANT.csv"):
            miss_for.append(sid)
    check(f"AUDIO.wav present", not miss_wav, f"missing {miss_wav[:10]}")
    check(f"TRANSCRIPT.csv present", not miss_txt, f"missing {miss_txt[:10]}")
    check(f"COVAREP.csv present", not miss_cov, f"missing {miss_cov[:10]}")
    check(f"FORMANT.csv present", not miss_for, f"missing {miss_for[:10]}")

    print("\n5. Audio properties")
    srs, durs, chans = set(), [], set()
    for sid in dirs:
        p = f"{ROOT}/{sid}_P/{sid}_AUDIO.wav"
        if not os.path.exists(p):
            continue
        with wave.open(p) as w:
            srs.add(w.getframerate())
            chans.add(w.getnchannels())
            durs.append(w.getnframes() / w.getframerate())
    check(f"sampling rate is 16 kHz throughout", srs == {16000}, f"found {srs}")
    check(f"all recordings are mono", chans == {1}, f"found {chans}")
    if durs:
        print(f"       interview length: {sum(durs)/3600:.1f} h in total, "
              f"{min(durs)/60:.1f}-{max(durs)/60:.1f} min each, "
              f"median {np.median(durs)/60:.1f} min")

    print("\n6. Transcripts")
    no_ellie_found, spk_odd = [], {}
    for sid in dirs:
        p = f"{ROOT}/{sid}_P/{sid}_TRANSCRIPT.csv"
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception as e:
            spk_odd[sid] = f"read failed: {e}"
            continue
        spk = set(df["speaker"].dropna().unique()) if "speaker" in df else set()
        if "Ellie" not in spk:
            no_ellie_found.append(sid)
        weird = spk - {"Ellie", "Participant"}
        if weird:
            spk_odd[sid] = sorted(weird)
    check(f"sessions without Ellie's transcript == the officially declared 451/458/480",
          set(no_ellie_found) == NO_ELLIE, f"found {sorted(no_ellie_found)}")
    check("the speaker field contains only Ellie / Participant", not spk_odd,
          f"unexpected {dict(list(spk_odd.items())[:5])}")

    print("\n7. COVAREP dimensionality")
    dims = set()
    for sid in dirs[:20]:
        p = f"{ROOT}/{sid}_P/{sid}_COVAREP.csv"
        if os.path.exists(p):
            with open(p) as f:
                dims.add(len(f.readline().split(",")))
    check(f"COVAREP has 74 dimensions (20 sessions sampled)", dims == {74}, f"found {dims}")

    print("\n8. Sessions with known defects are present (preprocessing corrects each of them)")
    for name, ids in [("timestamp offset", MISALIGNED), ("interrupted", INTERRUPT),
                      ("no Ellie transcript", NO_ELLIE)]:
        check(f"{name}: {sorted(ids)} all present", ids <= set(dirs))

    print("=" * 74)
    print("Result: " + ("all checks passed; the data is complete and usable" if ok
                        else "problems found; see the FAIL items above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
