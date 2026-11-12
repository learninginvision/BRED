# Adding a task

A task is a baseline program, a short description and a task config. This guide uses a task
called `MyTask`; replace the name throughout.

```text
resources/baseline/MyTask/
├── config.yaml          # name, description, domain, where the code lives
└── code/
    ├── experiment.py    # the baseline: trains, evaluates, writes metrics.json
    ├── metrics.json     # the baseline's verified result (the floor)
    └── ...              # helper modules and small data files
tasks/MyTask/
└── config.yaml          # time limit and metric
```

## 1. The baseline program

`resources/baseline/MyTask/code/` is the starting workspace. Each candidate works on a fresh copy of
it (or of the Champion's workspace), and the loop then runs

```bash
cd <workspace> && <code_runner.python> experiment.py
```

with `CUDA_VISIBLE_DEVICES` set to the candidate's GPU and the task's `code_timeout` as the time
limit. The program must:

- **Write `metrics.json` in the working directory** when it succeeds, as a flat JSON object that
  contains the primary metric, for example `{"best_acc": 0.9163}`. A run without a valid
  `metrics.json` counts as failed.
- **Exit with a non-zero code on failure.**
- **Run the full, frozen evaluation.** Candidates may change the model and the training recipe, so
  keep the evaluation protocol (data split, metric, number of seeds) fixed and visible in the code.

The copy leaves out symlinks, run outputs (`logs/`, `output/`, `checkpoints/`, `ckpt/`, `runs/`,
`wandb/`, `cache/`, `__pycache__/`), weights and logs (`.pth`, `.pt`, `.ckpt`, `.log`) and result
files (`metrics.json`, `final_info.json`, `result.txt`). Small data files can live inside `code/`
(AutoTSF keeps `data/ETTh1.csv` there); large datasets should stay outside the repository and be
read from absolute paths such as `/datasets/MyData/`.

## 2. The floor: `code/metrics.json`

Run the baseline once and save its `metrics.json` next to `experiment.py`. The loop does not re-run
the baseline: it reads this file as the floor that the first Champion has to beat, and refuses to
start if the file or the primary metric is missing.

## 3. `resources/baseline/MyTask/config.yaml`

```yaml
name: MyTask

task_description: |
  Your task is to improve <what> on <dataset>. The provided baseline <does what>.
  Focus on <the part candidates may change>. The evaluation metric is <metric> on <split>,
  higher is better.

domain: <research area, e.g. image classification>

relative_code_path: resources/baseline/MyTask/code
```

The prompts quote `task_description`, so state what may change and what must not.

## 4. `tasks/MyTask/config.yaml`

```yaml
code_timeout: 7200            # seconds per experiment run

metrics_format:               # keys expected in metrics.json
  best_acc: float

experiment_metric: |
  The experiment must report best_acc (accuracy on the dev set).

metric_explanation:
  primary: best_acc           # the metric the Champion is ranked by
  higher_better: true
```

## 5. Run it

```bash
python scripts/run_pipeline.py --task MyTask --rounds 2 --skip-experiments   # ideas only
python scripts/run_pipeline.py --task MyTask --rounds 10 --workers 2
```

Start with a short `code_timeout` and a small dataset to check that candidates run end to end,
then scale up.
