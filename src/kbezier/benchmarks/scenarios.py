"""
CL Scenarios: Task-IL, Class-IL, Domain-IL.

These wrappers control how task identity is used at test time
and how the model head is managed.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional


class Scenario(Enum):
    """Continual learning scenario types."""
    TASK_IL = "task-IL"       # Task-id available at test time, multi-head or masked
    CLASS_IL = "class-IL"     # No task-id at test, single expanding head
    DOMAIN_IL = "domain-IL"   # Same classes, different domains (e.g. permutations)


def get_scenario(name: str) -> Scenario:
    """Parse scenario name string to enum."""
    name_lower = name.lower().replace("_", "-")
    for s in Scenario:
        if s.value == name_lower:
            return s
    raise ValueError(f"Unknown scenario: {name}. Available: {[s.value for s in Scenario]}")


def get_classes_for_task(
    scenario: Scenario,
    task_id: int,
    task_classes: List[int],
    all_seen_classes: List[int],
) -> dict:
    """
    Get evaluation configuration for a task under the given scenario.

    Returns:
        Dict with keys:
            - eval_classes: which classes to evaluate on
            - use_task_mask: whether to mask logits to task's classes
    """
    if scenario == Scenario.TASK_IL:
        return {
            "eval_classes": task_classes,
            "use_task_mask": True,  # mask logits to this task's classes
        }
    elif scenario == Scenario.CLASS_IL:
        return {
            "eval_classes": all_seen_classes,
            "use_task_mask": False,  # evaluate over all seen classes
        }
    elif scenario == Scenario.DOMAIN_IL:
        return {
            "eval_classes": task_classes,  # same classes for all tasks
            "use_task_mask": False,
        }
    else:
        raise ValueError(f"Unhandled scenario: {scenario}")


def should_expand_head(scenario: Scenario) -> bool:
    """Whether the model head should grow with new tasks."""
    return scenario == Scenario.CLASS_IL


def needs_task_id_at_test(scenario: Scenario) -> bool:
    """Whether task identity is available at test time."""
    return scenario == Scenario.TASK_IL
