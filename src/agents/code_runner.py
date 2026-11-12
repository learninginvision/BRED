"""Code Runner: implement one idea with a coding agent, run it once and read its metric.

Each candidate gets a fresh copy of the starting workspace (the baseline or the Champion). The
coding agent (Claude Code through the Claude Agent SDK) edits the copy, then ``experiment.py`` is
run once under the task's time limit. A candidate succeeds when it writes a ``metrics.json`` with
a finite primary metric; there is no repair step in this release.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.config import Task
from src.memory import Candidate

logger = logging.getLogger(__name__)

# Run outputs that are not copied into a new workspace.
SKIP_DIRS = {"__pycache__", ".git", "logs", "output", "outputs", "runs", "wandb", "ckpt",
             "checkpoint", "checkpoints", "cache", ".cache", ".ipynb_checkpoints"}
SKIP_FILES = {"metrics.json", "final_info.json", "result.txt", "traceback.log", "run.log"}
SKIP_SUFFIXES = {".pth", ".pt", ".ckpt", ".pyc", ".log", ".tmp"}

PROMPT = """You are working in a machine learning experiment workspace (the current directory).
Implement the idea below by editing the code, mainly experiment.py.

## Task
{description}

## Evaluation
{metric_text}

## Idea: {title}
{idea}

## Rules
- Keep the data, the data split and the evaluation protocol unchanged.
- `python experiment.py` must still run without arguments and write metrics.json with the keys
  {keys}.
- Do not run the full training. Quick syntax or import checks are fine.

When you are done, reply with one sentence describing the change."""


@dataclass
class RunnerSettings:
    python: str = "python"
    allowed_tools: List[str] = field(default_factory=lambda: ["Read", "Write", "Edit", "Bash",
                                                              "Glob", "Grep"])
    claude_env: Dict[str, str] = field(default_factory=dict)
    agent_timeout: int = 1800
    max_turns: int = 60

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RunnerSettings":
        env = {k: str(v) for k, v in (cfg.get("claude_sdk_env") or {}).items() if v not in (None, "")}
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        if env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_AUTH_TOKEN"] = ""  # make the CLI use the API key, not a login token
        return cls(
            python=str(cfg.get("python") or "python"),
            allowed_tools=list(cfg.get("allowed_tools") or cls().allowed_tools),
            claude_env=env,
            agent_timeout=int(cfg.get("agent_timeout_sec", 1800)),
            max_turns=int(cfg.get("max_turns", 60)),
        )


# ---- workspace ------------------------------------------------------------

def copy_workspace(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: List[str]) -> List[str]:
        return [n for n in names
                if n in SKIP_DIRS or n in SKIP_FILES or Path(n).suffix in SKIP_SUFFIXES]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore, symlinks=False)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*.py")):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def read_metric(workspace: Path, key: str) -> tuple[Optional[float], Dict[str, Any]]:
    path = workspace / "metrics.json"
    if not path.is_file():
        return None, {}
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        value = float(metrics[key])
    except (ValueError, KeyError, TypeError):
        return None, {}
    return (value if math.isfinite(value) else None), metrics


# ---- coding agent -----------------------------------------------------------

async def claude_code(settings: RunnerSettings, prompt: str, workspace: Path) -> str:
    """One Claude Code session in ``workspace``; returns the agent's final text."""
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                                  ResultMessage, TextBlock)

    options = ClaudeAgentOptions(
        allowed_tools=settings.allowed_tools,
        permission_mode="acceptEdits",
        cwd=str(workspace),
        env=settings.claude_env or {},
        max_turns=settings.max_turns,
        setting_sources=["project", "local"],
    )
    parts: List[str] = []

    async def _session() -> None:
        async with ClaudeSDKClient(options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    parts.extend(b.text for b in message.content if isinstance(b, TextBlock))
                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        raise RuntimeError(f"coding agent error: {message.result}")
                    if message.result:
                        parts.append(str(message.result))
                    break

    await asyncio.wait_for(_session(), timeout=settings.agent_timeout)
    return parts[-1] if parts else ""


CodingAgent = Callable[[RunnerSettings, str, Path], Awaitable[str]]


# ---- execution ----------------------------------------------------------------

async def execute(settings: RunnerSettings, workspace: Path, gpu: Optional[str], timeout: int) -> int:
    """Run ``python experiment.py`` in ``workspace``; output goes to ``run.log``."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(workspace / "run.log", "wb") as log:
        proc = await asyncio.create_subprocess_exec(
            settings.python, "experiment.py", cwd=str(workspace), env=env,
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        try:
            return await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            return -9


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


async def run_candidate(settings: RunnerSettings, task: Task, candidate: Candidate,
                        base_dir: Path, gpu: Optional[str],
                        agent: CodingAgent = claude_code) -> Candidate:
    workspace = Path(candidate.workspace)
    copy_workspace(base_dir, workspace)
    before = _digest(workspace)

    prompt = PROMPT.format(description=task.description, metric_text=task.metric_text,
                           title=candidate.title, idea=candidate.idea,
                           keys=", ".join(task.metric_keys))
    try:
        summary = await agent(settings, prompt, workspace)
        logger.info("[%s] implemented: %s", candidate.cid, summary[:200])
    except Exception as exc:
        candidate.status, candidate.error = "failed", f"coding agent: {type(exc).__name__}: {exc}"
        return candidate
    if _digest(workspace) == before:
        candidate.status, candidate.error = "failed", "the coding agent did not change the code"
        return candidate

    code = await execute(settings, workspace, gpu, task.timeout)
    value, metrics = read_metric(workspace, task.primary)
    if code == 0 and value is not None:
        candidate.status, candidate.metric, candidate.metrics = "success", value, metrics
    else:
        reason = "timeout" if code == -9 else f"exit code {code}" if code else "no valid metric"
        candidate.status = "failed"
        candidate.error = f"{reason}\n{_tail(workspace / 'run.log')}"
    return candidate
