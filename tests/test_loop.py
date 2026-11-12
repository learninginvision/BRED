"""Offline tests: a toy task, a fake chat model and a fake coding agent (no network, no GPU)."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.code_runner import RunnerSettings, copy_workspace, run_candidate  # noqa: E402
from src.agents.controller import ScheduleController  # noqa: E402
from src.config import list_tasks, load_task  # noqa: E402
from src.llm import parse_json  # noqa: E402
from src.loop import run_loop  # noqa: E402
from src.memory import Candidate, Decision, Memory  # noqa: E402

TOY_EXPERIMENT = """import json
VALUE = 0.50
with open("metrics.json", "w") as f:
    json.dump({"acc": VALUE}, f)
"""


def make_toy_task(root: Path, experiment: str = TOY_EXPERIMENT) -> None:
    code = root / "resources" / "baseline" / "Toy" / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text(experiment)
    (code / "metrics.json").write_text(json.dumps({"acc": 0.50}))
    (root / "resources" / "baseline" / "Toy" / "config.yaml").write_text(
        "name: Toy\ntask_description: Improve acc.\ndomain: toy\n"
        "relative_code_path: resources/baseline/Toy/code\n")
    (root / "tasks" / "Toy").mkdir(parents=True)
    (root / "tasks" / "Toy" / "config.yaml").write_text(
        "code_timeout: 60\nmetrics_format:\n  acc: float\nexperiment_metric: acc\n"
        "metric_explanation:\n  primary: acc\n  higher_better: true\n")


class FakeLLM:
    async def chat_json(self, system: str, user: str):
        if "search queries" in system:
            return {"query": "toy query"}
        return {"ideas": [{"title": f"idea {i}", "description": f"raise VALUE by {i}"}
                          for i in range(1, 4)]}


async def fake_search(query, top_k, timeout):
    return [{"id": "0000.00000", "title": "A paper", "abstract": "An abstract."}]


async def fake_agent(settings, prompt, workspace: Path) -> str:
    """Raise VALUE by 0.01 times the idea number."""
    step = int(re.search(r"raise VALUE by (\d+)", prompt).group(1))
    path = workspace / "experiment.py"
    path.write_text(re.sub(r"VALUE = ([0-9.]+)",
                           lambda m: f"VALUE = {float(m.group(1)) + 0.01 * step:.2f}",
                           path.read_text()))
    return "raised VALUE"


def test_full_loop_on_toy_task(tmp_path):
    make_toy_task(tmp_path)
    task = load_task("Toy", tmp_path)
    run_dir = tmp_path / "run"
    memory = asyncio.run(run_loop(
        task, FakeLLM(), RunnerSettings(python=sys.executable), run_dir,
        rounds=3, workers=2, search=fake_search, agent=fake_agent))

    assert [r.decision.action for r in memory.rounds] == ["new line", "refine", "extend"]
    assert memory.champion is not None
    # Round 1: 0.50 + 0.02 = 0.52; each later round builds on the Champion.
    assert memory.champion.metric == pytest.approx(0.56)
    assert all(c.status == "success" for r in memory.rounds for c in r.candidates)
    saved = Memory.load(run_dir / "memory.json")
    assert saved.champion.cid == memory.champion.cid
    assert (run_dir / "rounds" / "round_1.json").is_file()


def test_skip_experiments(tmp_path):
    make_toy_task(tmp_path)
    task = load_task("Toy", tmp_path)
    memory = asyncio.run(run_loop(
        task, FakeLLM(), RunnerSettings(python=sys.executable), tmp_path / "run",
        rounds=2, workers=2, search=fake_search, agent=fake_agent, skip_experiments=True))
    assert memory.champion is None
    assert all(c.status == "skipped" for r in memory.rounds for c in r.candidates)


def test_unchanged_code_and_crash_fail(tmp_path):
    make_toy_task(tmp_path)
    task = load_task("Toy", tmp_path)
    settings = RunnerSettings(python=sys.executable)

    async def no_edit(settings, prompt, workspace):
        return "nothing"

    async def crash(settings, prompt, workspace):
        (workspace / "experiment.py").write_text("raise SystemExit(3)\n")
        return "broke it"

    for agent, expected in ((no_edit, "did not change"), (crash, "exit code 3")):
        cand = Candidate(cid="c", title="t", idea="i", workspace=str(tmp_path / agent.__name__))
        out = asyncio.run(run_candidate(settings, task, cand, task.code_dir, None, agent=agent))
        assert out.status == "failed" and expected in out.error


def test_record_keeps_champion_unless_strictly_better():
    memory = Memory(task="T", floor=0.5, higher_better=False)
    d = Decision("explore", "baseline", "x")
    memory.record(1, d, [Candidate("a", "a", "", status="success", metric=0.4)])
    assert memory.champion.cid == "a"
    memory.record(2, d, [Candidate("b", "b", "", status="success", metric=0.4),
                         Candidate("c", "c", "", status="failed")])
    assert memory.champion.cid == "a"
    memory.record(3, d, [Candidate("d", "d", "", status="success", metric=0.3)])
    assert memory.champion.cid == "d"


def test_controller_starts_from_baseline_without_champion():
    memory = Memory(task="T", floor=0.5, higher_better=True)
    decision = ScheduleController().decide(memory, 4, 10)
    assert (decision.mode, decision.base) == ("explore", "baseline")


def test_copy_workspace_skips_results(tmp_path):
    src = tmp_path / "src"
    (src / "logs").mkdir(parents=True)
    (src / "experiment.py").write_text("pass\n")
    (src / "metrics.json").write_text("{}")
    (src / "model.pth").write_text("x")
    copy_workspace(src, tmp_path / "dst")
    assert sorted(p.name for p in (tmp_path / "dst").iterdir()) == ["experiment.py"]


def test_parse_json():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure: {"ideas": []} done') == {"ideas": []}


@pytest.mark.parametrize("name", list_tasks(ROOT))
def test_bundled_tasks_load(name):
    task = load_task(name, ROOT)
    assert task.code_dir.is_dir() and (task.code_dir / "experiment.py").is_file()
    data = json.loads((task.code_dir / "metrics.json").read_text())
    assert task.primary in data
