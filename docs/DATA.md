# Task data

The repository holds code, configuration and small files only. The baseline code reads data from
`/datasets/...`, the convention of the InternAgent benchmark. To keep data elsewhere, create a
symlink or change the default path in that task's `experiment.py`.

| Task | Asset | Default path | Size | How to get it |
|---|---|---|---|---|
| AutoTSF | ETTh1.csv | `resources/baseline/AutoTSF/code/data/ETTh1.csv` | 2.6 MB | **Bundled** |
| AutoClsSST | SST-2: `train.tsv` (67,349), `dev.tsv` (872), `test.tsv` (1,821, labelled) | `/datasets/SST-2/` | 3.9 MB | **Bundled** in `resources/baseline/AutoClsSST/data/SST-2/`; copy it to the default path (below). Format `label<TAB>sentence`, no header. A few sentences differ from the copy on Hugging Face, so use the bundled files |
| AutoClsSST | BERT-base-uncased (weights and tokenizer) | `/datasets/BERT_ckpt/` | 421 MB | Hugging Face `google-bert/bert-base-uncased` (below) |
| AutoClsSST | RoBERTa-base (optional) | `/datasets/roberta-base/` | ~500 MB | Hugging Face `FacebookAI/roberta-base`; only needed by ideas that switch the encoder |
| AutoCls2D | CIFAR-100 (python version) | `/datasets/cifar100/` | 339 MB | Download once before the first run (below). Parallel candidates would otherwise all download into the same directory at once |
| AutoDepDAIC | DAIC-WOZ: audio and transcripts of 189 subjects, plus the AVEC2017 split files | `/datasets/DAIC-WOZ/` (override with `DAIC_ROOT`) | ~120 GB raw | **Licensed by USC ICT and not redistributable.** Request access at <https://dcapswoz.ict.usc.edu/>. Layout below |
| AutoDepDAIC | wav2vec2-base (frozen encoder) | `/datasets/wav2vec2-base/` (override with `W2V2_PATH`) | 363 MB | Hugging Face `facebook/wav2vec2-base` (below) |
| AutoDepDAIC | Per-layer wav2vec2 feature cache `features/w2v2_layers/{sid}.npy` | `$DAIC_ROOT/features/w2v2_layers/` | 3.6 GB | Built once by `run_baseline.sh` (below). Derived from licensed data, so not redistributable either |

One-time preparation:

```bash
# AutoClsSST: put the bundled SST-2 in place and download BERT-base
mkdir -p /datasets && cp -r resources/baseline/AutoClsSST/data/SST-2 /datasets/
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-uncased', local_dir='/datasets/BERT_ckpt', allow_patterns=['*.json', '*.txt', '*.safetensors'])"

# AutoCls2D: download CIFAR-100 (the directory must be writable)
python -c "from torchvision.datasets import CIFAR100; [CIFAR100('/datasets/cifar100', train=t, download=True) for t in (True, False)]"

# AutoDepDAIC: download wav2vec2-base
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-base', local_dir='/datasets/wav2vec2-base')"

# AutoDepDAIC: audit the data, build the feature cache, run the baseline once
# (about 50 minutes when the cache is empty; the argument is the GPU index)
CONDA_ENV=bred bash resources/baseline/AutoDepDAIC/run_baseline.sh 0
```

The AutoDepDAIC feature cache must exist before a run. The loop does not run the baseline (it reads
the floor from `code/metrics.json`), and without the cache every candidate of the first round would
try to build it at the same time.

## DAIC-WOZ layout

`$DAIC_ROOT` must be writable because the feature cache is written under it.

```text
$DAIC_ROOT/
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── full_test_split.csv
├── 300_P/   300_AUDIO.wav  300_TRANSCRIPT.csv  300_COVAREP.csv  300_FORMANT.csv  ...
├── 301_P/
└── ...
```

Each `{sid}_P.zip` of the official release is unzipped into a directory of the same name; the three
split files come from the same release.

- 188 subjects are used. Subject 440 is excluded because its raw files are corrupt; 342, 394, 398
  and 460 are gaps in the official numbering, not exclusions.
- Protocol: train, dev and test are pooled; subject-stratified 5-fold cross-validation over 5 seeds;
  the primary metric is macro-F1.
- `resources/baseline/AutoDepDAIC/audit_data.py` checks the data; `run_baseline.sh` calls it first.
