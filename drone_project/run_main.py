import os
import yaml
import time
import mavsdk
import asyncio
import numpy as np
import aioitertools
from mavsdk import System
from mavsdk import telemetry
# from stable_baselines3 import PPO
from mavsdk.camera import CameraError
# from drone_hover_env import DroneHoverEnv
from mavsdk.camera import Mode, Setting, Option
from mavsdk.mission import MissionItem, MissionPlan
# from stable_baselines3.common.policies import ActorCriticPolicy
from mavsdk.offboard import OffboardError, PositionNedYaw, Attitude

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14550")
    print("---Connecting to drone---")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("---Drone connected---")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Global position estimate is good")
            break

    print(f"telmetry: {dir(drone.telemetry.landed_state)}")
    async for state in drone.telemetry.landed_state():
        current_state = state
        print(f"current state is: {current_state}")
        break
    # initial position information
    # attitude_euler gives roll, pitch and yaw in degrees
    # position gives [lat_deg, lon_deg, abs_alt_m, rel_alt_m]
    # --> relative alt is meters above home position
    """lin pos"""
    async for lin_pos in drone.telemetry.position():
        lat1 = lin_pos.latitude_deg
        lon1 = lin_pos.longitude_deg
        alt1 = lin_pos.relative_altitude_m # or absolute_altitude_m
        break
    """"""
    """lin pos body -- reference to home"""
    async for lin_pos_body in drone.telemetry.position_velocity_ned():
        north = lin_pos_body.position.north_m
        east = lin_pos_body.position.east_m
        down = lin_pos_body.position.down_m 
        break
    """"""
    """lin vel"""
    async for lin_vel in drone.telemetry.position_velocity_ned():
        north_vel = lin_vel.velocity.north_m_s
        east_vel = lin_vel.velocity.east_m_s
        down_vel = lin_vel.velocity.down_m_s # or absolute_altitude_m
        break
    """"""
    """ang pos"""
    async for ang_pos in drone.telemetry.attitude_euler():
        roll = ang_pos.roll_deg
        pitch = ang_pos.pitch_deg
        yaw = ang_pos.yaw_deg
        break
    """"""
    """ang vel"""
    async for ang_vel in drone.telemetry.attitude_angular_velocity_body():
        roll_rate = ang_vel.roll_rad_s
        pitch_rate = ang_vel.pitch_rad_s
        yaw_rate = ang_vel.yaw_rad_s
        break
    """"""
    """quaternion"""
    # returns [w,x,y,z]
    async for quat in drone.telemetry.attitude_quaternion():
        """
        All rotations and axis systems follow the right-hand rule. The Hamilton quaternion product definition is
        used. A zero-rotation quaternion is represented by (1,0,0,0). The quaternion could also be written as 
        w + xi + yj + zk.
        For more info see: https://en.wikipedia.org/wiki/Quaternion
        """
        w = quat.w
        x = quat.x
        y = quat.y
        z = quat.z
        break
    """"""
    """home position"""
    async for home in drone.telemetry.home():
        break
    """"""


    
    # flight
    print("---Arming Drone---")
    await drone.action.arm()
    await asyncio.sleep(1)
    
    #cannot sleep > 2 since it has to takeoff within a few seconds of arming
    print("---Taking off---")
    await drone.action.takeoff()
    await asyncio.sleep(1)
    async for attitude in drone.telemetry.position():
        lat2 = attitude.latitude_deg
        lon2 = attitude.longitude_deg
        alt2 = attitude.absolute_altitude_m
        diff = alt2 - alt1
        print(f"difference in alt: {diff}")
        break

        
    await drone.offboard.set_attitude(Attitude(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        thrust_value=0.5
    ))

    # start offboard mode with above set params (network will then change this in a while loop)
    await drone.offboard.start()
    await asyncio.sleep(2)
    async for attitude in drone.telemetry.position():
        lat3 = attitude.latitude_deg
        lon3 = attitude.longitude_deg
        alt3 = attitude.absolute_altitude_m
        diff = alt3 - alt2
        print(f"difference in alt: {diff}")
        break

    # return to launch point
    await drone.action.return_to_launch()
    await asyncio.sleep(10)
    print("---Landing---")
    await drone.action.land()

def load_yaml_config(file_path):
    """Load the YAML configuration file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    asyncio.run(run())