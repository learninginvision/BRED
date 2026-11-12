#!/usr/bin/env python
"""Run the research loop on one task.

    python scripts/run_pipeline.py --task AutoTSF --rounds 10 --workers 3
    python scripts/run_pipeline.py --task AutoTSF --rounds 2 --skip-experiments
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.code_runner import RunnerSettings  # noqa: E402
from src.config import list_tasks, load_settings, load_task  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.loop import run_loop  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list_tasks(ROOT))
    p.add_argument("--rounds", type=int, default=10, help="round budget (default 10)")
    p.add_argument("--workers", type=int, default=3, help="candidates per round (default 3)")
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU ids, overrides code_runner.gpus; 'none' disables")
    p.add_argument("--run-name", default=None, help="output folder name (default: a timestamp)")
    p.add_argument("--skip-experiments", action="store_true",
                   help="decide, search and draft ideas, but do not run the coding agent")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    task = load_task(args.task, ROOT)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "storage" / "runs" / task.name / run_name
    if run_dir.exists():
        print(f"{run_dir} already exists; choose another --run-name", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")],
    )
    for noisy in ("httpx", "openai", "claude_agent_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    runner_cfg = settings.get("code_runner") or {}
    if args.gpus is None:
        gpus = runner_cfg.get("gpus") or []
    elif args.gpus.lower() == "none":
        gpus = []
    else:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if gpus and args.workers > len(gpus):
        logging.warning("--workers %d > %d GPUs: candidates will queue for a free GPU",
                        args.workers, len(gpus))

    memory = asyncio.run(run_loop(
        task=task,
        llm=LLM(settings["llm"]),
        runner=RunnerSettings.from_config(runner_cfg),
        run_dir=run_dir,
        rounds=args.rounds,
        workers=args.workers,
        gpus=gpus,
        retrieval=settings.get("retrieval"),
        skip_experiments=args.skip_experiments,
    ))
    champ = memory.champion
    print(f"\nRun folder: {run_dir}")
    print(f"Baseline {task.primary}: {memory.floor:.4f}")
    if champ:
        print(f"Champion {champ.cid} ({champ.title}): {champ.metric:.4f}, workspace {champ.workspace}")
    else:
        print("No candidate beat the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
