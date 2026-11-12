# Third-party notices

The code in this repository is released under the [MIT License](LICENSE), except for the parts
listed below.

## Baseline code from InternAgent

The baseline programs of three tasks are adapted from the task code of
[InternAgent](https://github.com/InternScience/InternAgent):

- `resources/baseline/AutoTSF/code/`
- `resources/baseline/AutoClsSST/code/`
- `resources/baseline/AutoCls2D/code/`

Copyright 2026 InternAgent Team, Shanghai Artificial Intelligence Laboratory. Licensed under the
Apache License, Version 2.0; the full text is in
[`resources/baseline/LICENSE-InternAgent`](resources/baseline/LICENSE-InternAgent). The BRED
authors modified these files, for example their data paths, output location, training schedule and
logging.

## Data and models used at run time

ETTh1 and SST-2 are bundled with this repository; the other assets are downloaded separately (see
[docs/DATA.md](docs/DATA.md)). All of them keep their own terms.

| Asset | Terms |
|---|---|
| ETTh1 (bundled with AutoTSF) | As released with the ETDataset |
| SST-2 (bundled with AutoClsSST) | As released with the Stanford Sentiment Treebank (Socher et al., 2013). The train and dev splits follow GLUE SST-2 (67,349 and 872 examples); the test split is the treebank's labelled one (1,821 sentences), not GLUE's unlabelled test set |
| BERT-base-uncased, RoBERTa-base, wav2vec2-base | Their Hugging Face model licences |
| DAIC-WOZ | USC ICT end-user license agreement; not redistributable |
| arXiv API | arXiv API terms of use |
