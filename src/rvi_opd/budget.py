from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetLedger:
    """Three non-interchangeable resource ledgers used in every result row."""

    teacher_generated_tokens: int = 0
    supervised_positions: int = 0
    teacher_forward_tokens: int = 0

    def add(self, other: BudgetLedger) -> None:
        self.teacher_generated_tokens += other.teacher_generated_tokens
        self.supervised_positions += other.supervised_positions
        self.teacher_forward_tokens += other.teacher_forward_tokens

    def validate(self) -> None:
        if (
            min(
                self.teacher_generated_tokens,
                self.supervised_positions,
                self.teacher_forward_tokens,
            )
            < 0
        ):
            raise ValueError("budget counts must be non-negative")


def paired_repair_span(
    trigger_position: int, bridge_token_count: int, trajectory_length: int
) -> range:
    """Return the unchanged-context repair window paired to one realised bridge.

    The repair arm keeps the original student trajectory and supervises the same
    number of positions as the bridge contains teacher-owned tokens. A truncated
    window must be reported and excluded from strict per-state budget matching.
    """

    if trigger_position < 0 or bridge_token_count < 0 or trajectory_length < 0:
        raise ValueError("positions and lengths must be non-negative")
    if trigger_position > trajectory_length:
        raise ValueError("trigger_position exceeds trajectory_length")
    return range(trigger_position, min(trigger_position + bridge_token_count, trajectory_length))
