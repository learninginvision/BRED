"""The round loop: Decide -> Generate -> Run -> Record."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.agents.code_runner import CodingAgent, RunnerSettings, claude_code, read_metric, run_candidate
from src.agents.controller import ScheduleController
from src.agents.idea_forge import draft_ideas
from src.agents.planner import plan_query
from src.agents.retriever import search_arxiv
from src.config import Task
from src.llm import LLM
from src.memory import Candidate, Memory

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int, int], Awaitable[List[Dict[str, str]]]]


class GPUPool:
    """Hands out one GPU per running candidate; without GPUs it only limits concurrency."""

    def __init__(self, gpus: Sequence[Any], workers: int):
        self.gpus = [str(g) for g in gpus]
        self._queue: asyncio.Queue = asyncio.Queue()
        for g in self.gpus:
            self._queue.put_nowait(g)
        self._sem = asyncio.Semaphore(max(1, workers))

    @contextlib.asynccontextmanager
    async def acquire(self):
        async with self._sem:
            if not self.gpus:
                yield None
                return
            gpu = await self._queue.get()
            try:
                yield gpu
            finally:
                self._queue.put_nowait(gpu)


def baseline_floor(task: Task) -> float:
    value, _ = read_metric(task.code_dir, task.primary)
    if value is None:
        raise FileNotFoundError(
            f"{task.code_dir / 'metrics.json'} must contain the baseline's '{task.primary}'")
    return value


async def run_loop(
    task: Task,
    llm: LLM,
    runner: RunnerSettings,
    run_dir: Path,
    rounds: int,
    workers: int,
    gpus: Sequence[Any] = (),
    retrieval: Optional[Dict[str, Any]] = None,
    skip_experiments: bool = False,
    search: SearchFn = search_arxiv,
    agent: CodingAgent = claude_code,
) -> Memory:
    retrieval = retrieval or {}
    top_k, timeout = int(retrieval.get("top_k", 10)), int(retrieval.get("timeout_sec", 30))
    memory = Memory(task=task.name, floor=baseline_floor(task), higher_better=task.higher_better)
    memory_path = run_dir / "memory.json"
    controller = ScheduleController()
    pool = GPUPool(gpus, workers)
    logger.info("task=%s floor %s=%.4f rounds=%d workers=%d",
                task.name, task.primary, memory.floor, rounds, workers)

    for t in range(1, rounds + 1):
        # 1. Decide
        decision = controller.decide(memory, t, rounds)
        logger.info("round %d: %s", t, decision.action)
        if decision.mode == "stop":
            break
        base_dir = task.code_dir if decision.base == "baseline" else Path(memory.champion.workspace)

        # 2. Generate
        query, papers = "", []
        if decision.mode == "explore":
            query = await plan_query(llm, task, decision)
            papers = await search(query, top_k, timeout)
            logger.info("round %d: query %r -> %d papers", t, query, len(papers))
        ideas = await draft_ideas(llm, task, decision, base_dir, papers, workers)
        logger.info("round %d: %d ideas", t, len(ideas))
        (run_dir / "rounds").mkdir(parents=True, exist_ok=True)
        (run_dir / "rounds" / f"round_{t}.json").write_text(json.dumps(
            {"decision": decision.__dict__, "base": str(base_dir), "query": query,
             "papers": papers, "ideas": ideas}, indent=2, ensure_ascii=False), encoding="utf-8")

        # 3. Run
        candidates = [Candidate(cid=f"r{t}_c{k}", title=i["title"], idea=i["description"],
                                workspace=str(run_dir / "workspaces" / f"r{t}_c{k}"))
                      for k, i in enumerate(ideas, 1)]
        if skip_experiments:
            for c in candidates:
                c.status = "skipped"
        else:
            async def _run(c: Candidate) -> Candidate:
                async with pool.acquire() as gpu:
                    logger.info("[%s] start on GPU %s: %s", c.cid, gpu, c.title)
                    return await run_candidate(runner, task, c, base_dir, gpu, agent=agent)

            candidates = list(await asyncio.gather(*(_run(c) for c in candidates)))

        # 4. Record
        entry = memory.record(t, decision, candidates)
        memory.save(memory_path)
        for c in candidates:
            logger.info("[%s] %s %s %s", c.cid, c.status,
                        "" if c.metric is None else f"{c.metric:.4f}", c.error.splitlines()[0] if c.error else "")
        logger.info("round %d: best %s, Champion %s (%.4f)%s", t, entry.best,
                    memory.champion.cid if memory.champion else "-", memory.best_so_far,
                    " NEW" if entry.new_champion else "")

    return memory
