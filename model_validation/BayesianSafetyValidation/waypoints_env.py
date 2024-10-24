from waypoint_handler import UpdatedWaypointHandler
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
from __future__ import annotations
from typing import Any, Literal
import numpy as np
from gymnasium import spaces
import gymnasium as gym


class Quadx_waypoints_2(QuadXWaypointsEnv):
    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        BSV_waypoints: None | np.ndarray = None,
    ):
        super().__init__(
            sparse_reward=sparse_reward,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.waypoints = UpdatedWaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=0.1,
            np_random=self.np_random,
            BSV_waypoints=BSV_waypoints
        )

        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
            }
        )
        self.sparse_reward = sparse_reward
        

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict(), BSV_waypoints: None | np.ndarray = None
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        print(f"BSV waypoints: {BSV_waypoints}")
        self.waypoints.reset(self.env, self.np_random, BSV_waypoints)
        self.info["num_targets_reached"] = 0
        super().end_reset()

        return self.state, self.info
    