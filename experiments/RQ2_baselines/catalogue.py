"""Frozen catalogue of trajectory-level baselines used by RQ2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    name: str
    filename: str
    family: str
    direction: int
    observability: str
    task_aware: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ``direction`` converts the stored quantity into a failure-risk score. The
# direction is fixed a priori: confidence metrics are negated, whereas larger
# uncertainty, instability, deviation, and distance values indicate more risk.
METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        "token_probability",
        "Token Probability",
        "Token_based_token_prob.json",
        "Token uncertainty",
        -1,
        "model internals",
    ),
    MetricSpec(
        "pcs",
        "Prediction Confidence Score (PCS)",
        "Token_based_pcs.json",
        "Token uncertainty",
        -1,
        "model internals",
    ),
    MetricSpec(
        "deepgini",
        "DeepGini",
        "Token_based_deepgini.json",
        "Token uncertainty",
        1,
        "model internals",
    ),
    MetricSpec(
        "entropy",
        "Entropy",
        "Token_based_entropy.json",
        "Token uncertainty",
        1,
        "model internals",
    ),
    MetricSpec(
        "action_position_instability",
        "Action Position Instability",
        "position_instability.json",
        "Action uncertainty",
        1,
        "VLA action stream",
    ),
    MetricSpec(
        "action_velocity_instability",
        "Action Velocity Instability",
        "velocity_instability.json",
        "Action uncertainty",
        1,
        "VLA action stream",
    ),
    MetricSpec(
        "action_acceleration_instability",
        "Action Acceleration Instability",
        "acceleration_instability.json",
        "Action uncertainty",
        1,
        "VLA action stream",
    ),
    MetricSpec(
        "execution_variability",
        "Execution Variability",
        "Execution_Variability.json",
        "Action uncertainty",
        1,
        "repeated VLA inference",
    ),
    MetricSpec(
        "tcp_position_instability",
        "TCP Position Instability",
        "TCP_position_instability.json",
        "Trajectory quality",
        1,
        "executed TCP trajectory",
    ),
    MetricSpec(
        "tcp_velocity_instability",
        "TCP Velocity Instability",
        "TCP_velocity_instability.json",
        "Trajectory quality",
        1,
        "executed TCP trajectory",
    ),
    MetricSpec(
        "tcp_acceleration_instability",
        "TCP Acceleration Instability",
        "TCP_acceleration_instability.json",
        "Trajectory quality",
        1,
        "executed TCP trajectory",
    ),
    MetricSpec(
        "trajectory_instability",
        "Trajectory Instability",
        "Trajectory_Instability_Gradients.json",
        "Trajectory quality",
        1,
        "executed TCP trajectory",
    ),
    MetricSpec(
        "optimal_trajectory",
        "Optimal Trajectory",
        "Optimal_Trajectory.json",
        "Task-aware quality",
        1,
        "object/goal state",
        task_aware=True,
    ),
)

METRIC_BY_ID = {spec.metric_id: spec for spec in METRIC_SPECS}
