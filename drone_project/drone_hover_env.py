from __future__ import annotations

from typing import Any, Literal

import numpy as np
import asyncio
import concurrent.futures
import threading

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
        # self.action = np.zeros((4,))
        self.action = np.zeros((4, ))
        self.lin_pos = np.zeros((3, ))
        self.alt = 0.0

        """ ACTION CONTINUOUS LOOP """
        self.offboard_task = self.loop.create_task(self._continuous_offboard_sender())
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        self.loop.run_forever()

    def run_async(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            print("---Async task timed out---")
            return None
        

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        # reset and step will use a blocking loop control method to ensure they execute
        print("***********************************RESET***********************************")
        print(f"reward was: {self.reward}")
        print(f"info is: {self.info}")
        print(f"altitude is: {self.lin_pos[2]}")
        options = dict()
        self.run_async(super().begin_reset(seed, options))
        self.run_async(super().end_reset(seed, options))
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
        # lin_pos, lin_vel, ang_pos, ang_vel, quaternion = self.loop.run_until_complete(super().compute_attitude())
        lin_pos, lin_vel, ang_pos, ang_vel, quaternion = self.run_async(super().compute_attitude())
        # combine everything
        self.alt = lin_pos[-1]
        self.lin_pos = lin_pos
        self.ang_vel = ang_vel
        self.ang_pos = ang_pos
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
            print("---max steps---")
            self.truncation |= True

        # if drone gets too high terminate the episode
        if self.alt > 50:
            print("---ALTITUDE---")
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

        # # exceed flight dome
        # if np.linalg.norm(self.lin_pos) > self.flight_dome_size:
        #     print("---OOB---")
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

        # reset the reward and set the action
        self.reward = 0.0
        self.run_async(self.set_action(self.action))       
        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1
        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        self.compute_base_term_trunc_reward()
        if not self.sparse_reward:
            yaw_rate_penalty = (abs(self.ang_vel[2])**2) * 0.01
            self.reward -= (yaw_rate_penalty)
            
            linear_distance = np.linalg.norm(self.lin_pos - np.array([self.lin_pos[0], self.lin_pos[1], 10.0]))
            angular_distance = np.linalg.norm(self.ang_pos[:2])
            self.reward -= linear_distance # + angular_distance
            self.reward += 1.0
