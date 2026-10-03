"""蓝队共享的轻量、不可变数据契约。"""

from dataclasses import dataclass
from typing import Optional, Tuple

WorldPoint = Tuple[float, float]


@dataclass(frozen=True)
class PoseEstimate:
    position: Optional[WorldPoint]
    heading: Optional[float]
    position_variance: float
    heading_variance: float
    source: str
    frame_id: str
    step: int
    valid: bool


@dataclass(frozen=True)
class NavigationGoal:
    goal_id: str
    kind: str
    position: WorldPoint
    tolerance: float
    priority: float
    created_step: int


@dataclass(frozen=True)
class PathPlan:
    status: str
    waypoints: Tuple[WorldPoint, ...]
    total_cost: float
    map_revision: int
    created_step: int
    reason: str
    geometric_length_px: float = 0.0
    clearance_px: float = 0.0
    goal_id: str = ""


@dataclass(frozen=True)
class ExplorationDecision:
    status: str
    goal: Optional[NavigationGoal]
    frontier_id: Optional[str]
    expected_gain: float
    estimated_cost: float
    lease_until_step: int
    reason: str


@dataclass(frozen=True)
class PeerExploreLease:
    peer_id: int
    frontier_id: str
    center: WorldPoint
    bid: float
    frame_id: str
    team_size: int
    epoch: int
    sender_step: int
    received_step: int
    expires_step: int


@dataclass(frozen=True)
class ExplorationInput:
    pose: PoseEstimate
    step: int
    world_size: Optional[WorldPoint]
    drone_id: int
    drone_count: Optional[int]
    has_bomb_task: bool
    carrying: bool
    peer_leases: Tuple[PeerExploreLease, ...]
