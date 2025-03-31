from __future__ import annotations

from typing import Any, Literal

import numpy as np
import asyncio

from base_env import BaseDroneEnv


class DroneHoverEnv(BaseDroneEnv):
    """Simple Hover Environment using PX4, Gazebo and MAVSDK

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.

    """

    def __init__(self, env_config):
        """
        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.

        """
        # Create a dedicated event loop for running async tasks synchronously.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Run async initialization code
        self.loop.run_until_complete(self.async_init(env_config))
        self.step_count = 0
        # sparse reward or not
        sparse_reward = env_config.get("sparse_reward", False)
        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward
        self.action = np.zeros((4,))

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        # reset and step will use a blocking loop control method to ensure they execute
        self.loop.run_until_complete(super().begin_reset(seed, options))
        self.loop.run_until_complete(super().end_reset(seed, options))
        self.compute_state()
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
        lin_pos, lin_vel, ang_pos, ang_vel, quaternion = self.loop.run_until_complete(super().compute_attitude())
        # combine everything
        if self.angle_representation == 0:
            self.state = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            self.state = np.concatenate(
                [
                    ang_vel, 
                    quaternion, 
                    lin_vel, 
                    lin_pos, 
                    self.action,
                ], 
                axis=-1,
            )

    def compute_term_trunc_reward(self) -> None:
        """compute_term_trunc_reward."""
        raise NotImplementedError

    def compute_base_term_trunc_reward(self) -> None:
        """compute_base_term_trunc_reward."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True

        # # if anything hits the floor, basically game over
        # if np.any(self.env.contact_array[self.env.planeId]):
        #     self.reward = -100.0
        #     self.info["collision"] = True
        #     self.termination |= True

        # # exceed flight dome
        # if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
        #     self.reward = -100.0
        #     self.info["out_of_bounds"] = True
        #     self.termination |= True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info

        """
        self.action = action.copy()
        print(f"action is: {self.action}")

        # reset the reward and set the action
        self.reward = -0.1
        self.loop.run_until_complete(self.set_action(self.action))
        

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break
            # self.loop.run_until_complete(self.compute_state)
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        self.compute_base_term_trunc_reward()
        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            # linear_distance = np.linalg.norm(
            #     self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
            # )
            # Negative Reward For High Yaw rate, To prevent high yaw while training
            # yaw_rate = abs(
            #     self.env.state(0)[0][2]
            #)  # Assuming z-axis is the last component
            # yaw_rate_penalty = 0.01 * yaw_rate**2  # Add penalty for high yaw rate
            # self.reward -= (
            #     yaw_rate_penalty  # You can adjust the coefficient (0.01) as needed
            # )

            # how far are we from 0 roll pitch
            # angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

            # self.reward -= linear_distance + angular_distance
            self.reward += 1.0