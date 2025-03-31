from __future__ import annotations

from typing import Any, Literal

import gymnasium
import numpy as np
from gymnasium import spaces

import os
import yaml
import time
import mavsdk
import asyncio
import numpy as np
import aioitertools
from mavsdk import System
from mavsdk import telemetry
from stable_baselines3 import PPO
from mavsdk.camera import CameraError
from mavsdk.camera import Mode, Setting, Option
from mavsdk.mission import MissionItem, MissionPlan
from stable_baselines3.common.policies import ActorCriticPolicy
from mavsdk.offboard import OffboardError, PositionNedYaw, Attitude

"""
GO BACK TO GPT AND SEE HOW TO SET UP A SYNCRONOUS WRAPPER FOR ASYNC ENVS
"""


class BaseDroneEnv(gymnasium.Env):
    """Base environment that exposes PX4 / mavsdk to the gym API"""

    async def async_init(self, env_config):
        agent_hz = env_config.get("agent_hz", 40)
        flight_mode = env_config.get("flight_mode", 0)
        flight_dome_size = env_config.get("flight_dome_size", 50)
        max_duration_seconds = env_config.get("max_duration_seconds", 45)
        start_pos = env_config.get("start_pos", np.array([[0.0, 0.0, 1.0]]))
        start_orn = env_config.get("start_orn", np.array([[0.0, 0.0, 0.0]]))
        angle_representation = env_config.get("angle_representation", "quaternion")
        
        """sim rate depends on GZ/PX4 -- look up"""
        sim_rate = 200
        
        if sim_rate % agent_hz != 0:
            lowest = int(sim_rate / (int(sim_rate / agent_hz) + 1))
            highest = int(sim_rate / int(sim_rate / agent_hz))
            raise ValueError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise ValueError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
        )

        # action space setup
        high = np.ones((4,))
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # the whole implicit state space = attitude + previous action + auxiliary 
        # information
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.action_space.shape[0],
            ),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(sim_rate / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

        # drone object
        self.drone = System()
        """By using await here, we are creating coroutines that will run with the main event loop
            allowing I/O operations to still happen for the drone, but these tasks will sequentially be done
            by this style of await task_1, await task_2, etc. 
            We can have them asynchronous by using asyncio.gather() or asyncio.create_task()
        """
        await self.connect_drone()
        await self.check_drone_status()
        self.home_position = await self.get_init_position()

    async def set_offboard_params(self):
        await self.drone.offboard.set_attitude(Attitude(
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            thrust_value=0.5
        ))
        print("--Set offboard params--")

    async def connect_drone(self):
        await self.drone.connect(system_address="udp://:14550")
        print("---Connecting to drone---")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("---Drone connected---")
                break

    async def check_drone_status(self):
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("---Global position estimate is good---")
                break

    async def get_init_position(self):
        async for home in self.drone.telemetry.home():
            lat = home.latitude_deg
            lon = home.longitude_deg
            alt = home.absolute_altitude_m
            position_np = np.array([lat, lon, alt])
            break 
        return position_np
        

    async def reset(self, *, seed, options):
        raise NotImplementedError

    async def begin_reset(self, seed: None | int = None, options: None | dict[str, Any] = dict()) -> None:

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        async for state in self.drone.telemetry.landed_state():
            options['current_state'] = state
            break
        if not options['current_state'] == "ON_GROUND":
            await self.drone.action.return_to_launch()
            await asyncio.sleep(1)
            print("---Landing---")
            await self.drone.action.land()

        

    async def end_reset(self, seed: None | int = None, options: None | dict[str, Any] = dict()) -> None:        
        # connect to a new drone object that connects to the sim
        """might have to break connection and reset it"""
        async for state in self.drone.core.connection_state():
            if not state.is_connected:
                print("---Drone connected---")
                self.drone = System()
                await self.drone.connect(system_address="udp://:14550")
                print("---Connecting to drone---")
                async for state in self.drone.core.connection_state():
                    if state.is_connected:
                        print("---Drone connected---")
                        break
                break
    
                async for health in self.drone.telemetry.health():
                    if health.is_global_position_ok and health.is_home_position_ok:
                        print("Global position estimate is good")
                        break
                print("---Arming Drone---")
                await self.drone.action.arm()
                await asyncio.sleep(1)
                print("---Taking off---")
                await self.drone.action.takeoff()
                await asyncio.sleep(5)
                await self.set_offboard_params()
                await self.drone.offboard.start()
            else:
                print("---Arming Drone---")
                await self.drone.action.arm()
                await asyncio.sleep(1)
                print("---Taking off---")
                await self.drone.action.takeoff()
                await asyncio.sleep(5)
                await self.set_offboard_params()
                await self.drone.offboard.start()
                break

    async def compute_state(self) -> None:
        """Computes the state of the QuadX."""
        raise NotImplementedError

    async def build_state(self):
        """Builds the observation state of the agent in a asynchronous manner using asyncio.gather()
        Args:
            -get_lin_pos: linear position in [lat_deg, lon_deg, alt_m]
            -get_lin_vel: linear velocity in NED meters/sec
            -get_ang_pos: angular position [roll_deg, pitch_deg, yaw_deg]
            -get_ang_vel: angular velocity in rad/sec
            -get_quat: quaternion in [w, x, y, z] with null rotation as [1, 0, 0, 0]
        Returns:
            -state of the agent
        """
        async def get_lin_pos():
            async for lin_pos in self.drone.telemetry.position():
                lat1 = lin_pos.latitude_deg
                lon1 = lin_pos.longitude_deg
                alt1 = lin_pos.relative_altitude_m
                return np.array([lat1, lon1, alt1])   

        async def get_lin_vel():
            async for lin_vel in self.drone.telemetry.position_velocity_ned():
                north_vel = lin_vel.velocity.north_m_s
                east_vel = lin_vel.velocity.east_m_s
                down_vel = lin_vel.velocity.down_m_s # or absolute_altitude_m
                return np.array([north_vel, east_vel, down_vel])

        async def get_ang_pos():
            async for ang_pos in self.drone.telemetry.attitude_euler():
                roll = ang_pos.roll_deg
                pitch = ang_pos.pitch_deg
                yaw = ang_pos.yaw_deg
                return np.array([roll, pitch, yaw])

        async def get_ang_vel():
            async for ang_vel in self.drone.telemetry.attitude_angular_velocity_body():
                roll_rate = ang_vel.roll_rad_s
                pitch_rate = ang_vel.pitch_rad_s
                yaw_rate = ang_vel.yaw_rad_s
                return np.array([roll_rate, pitch_rate, yaw_rate])

        async def get_quat():
            async for quat in self.drone.telemetry.attitude_quaternion():
                w = quat.w
                x = quat.x
                y = quat.y
                z = quat.z
                return np.array([w, x, y, z])

        lin_pos_np, lin_vel_np, ang_pos_np, ang_vel_np, quat_np = await asyncio.gather(
            get_lin_pos(),
            get_lin_vel(),
            get_ang_pos(),
            get_ang_vel(),
            get_quat(),
        )

        # state = np.array([lin_pos_np, lin_vel_np, ang_pos_np, ang_vel_np, quat_np])
        return lin_pos_np, lin_vel_np, ang_pos_np, ang_vel_np, quat_np

    async def compute_attitude(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        # build the raw state
        raw_state =  await self.build_state()

        # state breakdown
        lin_pos = raw_state[0]
        lin_vel = raw_state[1]
        ang_pos = raw_state[2]
        ang_vel = raw_state[3]
        quaternion = raw_state[4]

        return raw_state



    async def set_action(self, actions) -> None:
        """Sets the offboard actions [roll, pitch, yaw, thrust]
        Args:
            -actions: [-1:1, -1:1, -1:1, -1:1]
        """
        self.drone.offboard.set_actuator_control(actions)






        