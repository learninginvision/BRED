"""Search Planner: turn the round's instruction into one arXiv search query."""

from __future__ import annotations

import logging

from src.config import Task
from src.llm import LLM
from src.memory import Decision

logger = logging.getLogger(__name__)

SYSTEM = "You write concise search queries for finding relevant machine learning papers on arXiv."

USER = """Task: {description}
Research area: {domain}
Goal for this round: {instruction}

Write one arXiv search query of 3 to 8 words for papers that could help with this goal.
Reply as JSON: {{"query": "..."}}"""


async def plan_query(llm: LLM, task: Task, decision: Decision) -> str:
    try:
        reply = await llm.chat_json(SYSTEM, USER.format(
            description=task.description, domain=task.domain, instruction=decision.instruction))
        query = str(reply.get("query", "")).strip() if isinstance(reply, dict) else ""
    except Exception as exc:
        logger.warning("query planning failed (%s); using the task domain", exc)
        query = ""
    return query or task.domain or task.name
