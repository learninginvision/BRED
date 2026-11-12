"""Idea Forge: draft N ideas for the round in one LLM call."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.config import Task
from src.llm import LLM
from src.memory import Decision

logger = logging.getLogger(__name__)

MAX_CODE_CHARS = 20000
MAX_ABSTRACT_CHARS = 1200

SYSTEM = ("You are a machine learning researcher. You propose concrete, implementable changes "
          "to an existing experiment.")

USER = """## Task
{description}

## Evaluation
{metric_text}
Primary metric: {primary} ({direction} is better).

## Starting code (experiment.py)
```python
{code}
```

## Goal for this round
{instruction}
{papers}
Propose {n} different ideas for this goal. Each idea must be implementable by editing the code
above and must keep the evaluation protocol unchanged.

Reply as JSON: {{"ideas": [{{"title": "...", "description": "..."}}]}}"""


def read_code(workspace: Path) -> str:
    path = workspace / "experiment.py"
    if not path.is_file():
        return "(experiment.py not found)"
    code = path.read_text(encoding="utf-8", errors="replace")
    if len(code) > MAX_CODE_CHARS:
        code = code[:MAX_CODE_CHARS] + "\n# ... (truncated)"
    return code


def _format_papers(papers: List[Dict[str, str]]) -> str:
    if not papers:
        return ""
    lines = ["", "## Related papers"]
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract", "")[:MAX_ABSTRACT_CHARS]
        lines.append(f"{i}. {p.get('title', '')} (arXiv {p.get('id', '')})\n   {abstract}")
    return "\n".join(lines) + "\n"


async def draft_ideas(llm: LLM, task: Task, decision: Decision, base_dir: Path,
                      papers: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    user = USER.format(
        description=task.description,
        metric_text=task.metric_text,
        primary=task.primary,
        direction="higher" if task.higher_better else "lower",
        code=read_code(base_dir),
        instruction=decision.instruction,
        papers=_format_papers(papers),
        n=n,
    )
    try:
        reply = await llm.chat_json(SYSTEM, user)
    except Exception as exc:
        logger.error("idea drafting failed: %s", exc)
        return []
    items = reply.get("ideas", []) if isinstance(reply, dict) else reply
    ideas = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("description"):
            ideas.append({"title": str(item.get("title", "")).strip() or "untitled",
                          "description": str(item["description"]).strip()})
    return ideas[:n]
