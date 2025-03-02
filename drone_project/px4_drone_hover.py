import os
import time
import mavsdk
import asyncio
import numpy as np
import aioitertools
from mavsdk import telemetry
from mavsdk import System
from mavsdk.camera import CameraError
from mavsdk.camera import Mode, Setting, Option
from mavsdk.mission import MissionItem, MissionPlan
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

    # initial position information
    async for attitude in drone.telemetry.position():
        lat1 = attitude.latitude_deg
        lon1 = attitude.longitude_deg
        alt1 = attitude.absolute_altitude_m
        break
    
    # flight
    print("---Arming Drone---")
    await drone.action.arm()
    await asyncio.sleep(1)
    #cannot sleep > 2 since it has to takeoff within a few seconds of arming
    print("---Taking off---")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    async for attitude in drone.telemetry.position():
        lat2 = attitude.latitude_deg
        lon2 = attitude.longitude_deg
        alt2 = attitude.absolute_altitude_m
        diff = alt2 - alt1
        print(f"difference in alt: {diff}")
        break

    # neural network will control the roll/pitch/yaw/thrust values to learn how to hover at a point
    await drone.offboard.set_attitude(Attitude(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        thrust_value=0.5
    ))

    # start offboard mode with above set params (network will then change this in a while loop)
    await drone.offboard.start()
    await asyncio.sleep(20)
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

if __name__ == "__main__":
    asyncio.run(run())
