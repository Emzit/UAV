"""蓝队探索配置。数值为设计初值，后续用多回合评测调整。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationConfig:
    exploration_enabled: bool = True
    coverage_margin_px: float = 45.0
    coverage_lane_max_spacing_px: float = 160.0
    coverage_goal_radius_px: float = 30.0
    coverage_goal_hold_steps: int = 3
    coverage_stall_steps: int = 45
    coverage_min_progress_px: float = 10.0
    coverage_retry_steps: int = 120
    frontier_min_cluster_cells: int = 3
    frontier_refresh_steps: int = 20
    frontier_candidate_cap: int = 12
    frontier_match_radius_px: float = 60.0
    frontier_peer_match_px: float = 100.0
    frontier_information_radius_px: float = 200.0
    frontier_min_gain: float = 0.05
    goal_switch_margin: float = 0.12
    lease_ttl_steps: int = 60
    lease_refresh_steps: int = 20
    pose_variance_max_px2: float = 400.0
    pose_heading_variance_max_rad2: float = 0.04
    route_query_max_expansions: int = 800
    debug_log_every_steps: int = 100
