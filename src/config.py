"""Configuration loading: the global config, the task definition and the run directory."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Replace ``${VAR}`` with the value of the environment variable VAR (empty if unset)."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path | None = None) -> Dict[str, Any]:
    """Load ``config.yaml``. Variables in ``.env`` are loaded first; the shell environment wins."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    return _expand_env(load_yaml(path or ROOT / "config.yaml"))


@dataclass
class Task:
    """Everything the loop needs to know about one task."""

    name: str
    description: str
    domain: str
    background: str
    code_dir: Path            # the baseline workspace
    timeout: int              # seconds per experiment run
    metric_text: str          # how the task is evaluated, quoted to the models
    metric_keys: List[str]    # keys expected in metrics.json
    primary: str              # the metric the Champion is ranked by
    higher_better: bool


def list_tasks(root: Path = ROOT) -> List[str]:
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "config.yaml").is_file())


def load_task(name: str, root: Path = ROOT) -> Task:
    """Read ``resources/baseline/{name}/config.yaml`` and ``tasks/{name}/config.yaml``."""
    base = load_yaml(root / "resources" / "baseline" / name / "config.yaml")
    cfg = load_yaml(root / "tasks" / name / "config.yaml")
    explanation = cfg.get("metric_explanation") or {}
    primary = explanation.get("primary")
    if not primary:
        raise ValueError(f"tasks/{name}/config.yaml: metric_explanation.primary is required")
    code_dir = Path(base.get("relative_code_path") or f"resources/baseline/{name}/code")
    if not code_dir.is_absolute():
        code_dir = root / code_dir
    return Task(
        name=name,
        description=str(base.get("task_description", "")).strip(),
        domain=str(base.get("domain", "")).strip(),
        background=str(base.get("background", "")).strip(),
        code_dir=code_dir,
        timeout=int(cfg.get("code_timeout", 7200)),
        metric_text=str(cfg.get("experiment_metric", "")).strip(),
        metric_keys=list((cfg.get("metrics_format") or {primary: "float"}).keys()),
        primary=str(primary),
        higher_better=bool(explanation.get("higher_better", True)),
    )
