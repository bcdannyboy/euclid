from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class LedgerTask:
    task_id: str
    title: str
    files: tuple[str, ...]
    subtasks: tuple[str, ...]
    required_tests: tuple[str, ...]


def parse_master_ledger(path: Path) -> dict[str, LedgerTask]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks: dict[str, LedgerTask] = {}
    current_task_id: str | None = None
    current_title = ""
    current_files: list[str] = []
    current_subtasks: list[str] = []
    current_required_tests: list[str] = []
    section: str | None = None

    def flush() -> None:
        if current_task_id is None:
            return
        tasks[current_task_id] = LedgerTask(
            task_id=current_task_id,
            title=current_title,
            files=tuple(current_files),
            subtasks=tuple(current_subtasks),
            required_tests=tuple(current_required_tests),
        )

    for line in lines:
        task_match = re.match(r"^### Task ([0-9]{1,2}\.[0-9]{2}) — (.+)$", line)
        if task_match:
            flush()
            current_task_id = task_match.group(1)
            current_title = task_match.group(2).strip()
            current_files = []
            current_subtasks = []
            current_required_tests = []
            section = None
            continue
        if current_task_id is None:
            continue
        if line == "**Files**":
            section = "files"
            continue
        if line == "**Subtasks**":
            section = "subtasks"
            continue
        if line == "**Required tests**":
            section = "required_tests"
            continue
        if line.startswith("### Task ") or line == "---":
            section = None
            continue
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if section == "files":
            current_files.append(item)
        elif section == "subtasks":
            current_subtasks.append(item)
        elif section == "required_tests":
            current_required_tests.append(item)

    flush()
    return tasks


def flatten_plan_entries(tasks: dict[str, LedgerTask]) -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    for task_id, task in tasks.items():
        entries.add((task_id, "__task__"))
        for index, _subtask in enumerate(task.subtasks, start=1):
            entries.add((task_id, f"{task_id}.s{index:02d}"))
    return entries
