"""Research Controller (Decide).

This release ships a fixed schedule in place of the LLM controller described in the paper. It
returns the same kind of decision, so the rest of the loop is unchanged:

- no Champion yet: start a new line from the baseline;
- otherwise: alternate between extending the Champion with a new mechanism and refining it.

It never stops early; the run ends when the round budget is used.
"""

from __future__ import annotations

from src.memory import Decision, Memory

NEW_LINE = ("Propose a new method that improves the primary metric over the baseline. "
            "Keep the evaluation protocol unchanged.")
EXTEND = ("Add one new mechanism on top of the current best solution to improve the primary "
          "metric further.")
REFINE = ("Make small, incremental changes to the current best solution (hyperparameters, "
          "training schedule, regularisation) without adding a new mechanism.")


class ScheduleController:
    def decide(self, memory: Memory, round_no: int, total_rounds: int) -> Decision:
        if memory.champion is None:
            return Decision("explore", "baseline", NEW_LINE, "No Champion yet.")
        if round_no % 2 == 1:
            return Decision("explore", "champion", EXTEND, "Scheduled extension of the Champion.")
        return Decision("exploit", "champion", REFINE, "Scheduled refinement of the Champion.")
