from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


WORKFLOW_SKILL = "workflow-orchestrator"
ALGORITHM_SKILL = "algorithm-orchestrator"
REPORT_SKILL = "report-authoring"


@dataclass
class PlannerSkillRoute:
    name: str
    reason: str

    def to_state_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


class PlannerSkillRouter:
    def __init__(self, max_auto_skills: int = 2) -> None:
        self.max_auto_skills = max_auto_skills

    def route(
        self,
        turn_text: str,
        state: dict[str, Any],
        available_skill_names: Iterable[str],
    ) -> List[PlannerSkillRoute]:
        return []
