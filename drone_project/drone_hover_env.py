"""QuadX Hover Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from base_env import QuadXBaseEnv


class QuadXHoverEnv(QuadXBaseEnv):
    """Simple Hover Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash for the longest time possible.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        flight_mode: int = 0,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        if self.angle_representation == 0:
            self.state = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            self.state = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()
        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
            )
            # Negative Reward For High Yaw rate, To prevent high yaw while training
            yaw_rate = abs(
                self.env.state(0)[0][2]
            )  # Assuming z-axis is the last component
            yaw_rate_penalty = 0.01 * yaw_rate**2  # Add penalty for high yaw rate
            self.reward -= (
                yaw_rate_penalty  # You can adjust the coefficient (0.01) as needed
            )

            # how far are we from 0 roll pitch
            angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

            self.reward -= linear_distance + angular_distance
            self.reward += 1.0