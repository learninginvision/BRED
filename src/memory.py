"""The two persistent objects of a run: the Champion and the log of rounds."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    mode: str            # "explore" | "exploit" | "stop"
    base: str            # "baseline" | "champion"
    instruction: str
    rationale: str = ""

    @property
    def action(self) -> str:
        if self.mode == "stop":
            return "stop"
        if self.mode == "exploit":
            return "refine"
        return "new line" if self.base == "baseline" else "extend"


@dataclass
class Candidate:
    cid: str
    title: str
    idea: str
    workspace: str = ""
    status: str = "pending"       # "success" | "failed" | "skipped"
    metric: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class Champion:
    cid: str
    title: str
    workspace: str
    metric: float
    round: int


@dataclass
class RoundRecord:
    round: int
    decision: Decision
    candidates: List[Candidate]
    best: Optional[float]
    new_champion: bool


@dataclass
class Memory:
    task: str
    floor: float                  # the baseline metric the first Champion must beat
    higher_better: bool
    champion: Optional[Champion] = None
    rounds: List[RoundRecord] = field(default_factory=list)

    # ---- comparison -------------------------------------------------------
    def better(self, value: Optional[float], than: Optional[float]) -> bool:
        """Strict improvement in the task's metric direction."""
        if value is None or not math.isfinite(value):
            return False
        if than is None:
            return True
        return value > than if self.higher_better else value < than

    @property
    def best_so_far(self) -> float:
        return self.champion.metric if self.champion else self.floor

    # ---- record -----------------------------------------------------------
    def record(self, round_no: int, decision: Decision, candidates: List[Candidate]) -> RoundRecord:
        """Append a round; the best candidate becomes the Champion only on a strict improvement."""
        scored = [c for c in candidates if c.status == "success" and c.metric is not None]
        best = None
        for c in scored:
            if best is None or self.better(c.metric, best.metric):
                best = c
        new_champion = best is not None and self.better(best.metric, self.best_so_far)
        if new_champion:
            self.champion = Champion(cid=best.cid, title=best.title, workspace=best.workspace,
                                     metric=float(best.metric), round=round_no)
        entry = RoundRecord(round=round_no, decision=decision, candidates=candidates,
                            best=best.metric if best else None, new_champion=new_champion)
        self.rounds.append(entry)
        return entry

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        rounds = [
            RoundRecord(
                round=r["round"],
                decision=Decision(**r["decision"]),
                candidates=[Candidate(**c) for c in r["candidates"]],
                best=r.get("best"),
                new_champion=r.get("new_champion", False),
            )
            for r in d.get("rounds", [])
        ]
        champ = d.get("champion")
        return cls(task=d["task"], floor=d["floor"], higher_better=d["higher_better"],
                   champion=Champion(**champ) if champ else None, rounds=rounds)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Memory":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
