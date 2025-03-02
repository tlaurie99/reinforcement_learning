# import cv2
import os
import time
import mavsdk
import asyncio
import numpy as np
from mavsdk import System
from mavsdk.camera import CameraError
from mavsdk.camera import Mode, Setting, Option
from hloc_pipeline import main as hloc_pipeline
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.offboard import OffboardError, PositionNedYaw

async def run():
    drone = System()
    # print(dir(drone))
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

    # Select the camera if needed
    print("Selecting camera 1")
    try:
        await drone.camera.select_camera(1)
        print("Camera 1 selected successfully")
    except Exception as e:
        print(f"Failed to select camera 1: {str(e)}")

    mission_items = []

    mission_items.append(MissionItem(
        latitude_deg=47.398039859999997,
        longitude_deg=8.5455725400000002,
        relative_altitude_m=10,
        speed_m_s=2.2352,
        is_fly_through=True,
        gimbal_pitch_deg=float('nan'),
        gimbal_yaw_deg=0,
        camera_action=MissionItem.CameraAction.START_PHOTO_INTERVAL,
        loiter_time_s=5.0,
        camera_photo_interval_s=0.5,
        acceptance_radius_m=float('nan'),
        yaw_deg=float('nan'),
        camera_photo_distance_m=1.0,
        vehicle_action=MissionItem.VehicleAction.NONE,
    ))
    mission_items.append(MissionItem(
        latitude_deg=47.398036222362471,
        longitude_deg=8.5450146439425509,
        relative_altitude_m=5,
        speed_m_s=2.2352,
        is_fly_through=True,
        gimbal_pitch_deg=float('nan'),
        gimbal_yaw_deg=90,
        camera_action=MissionItem.CameraAction.NONE,
        loiter_time_s=float('nan'),
        camera_photo_interval_s=float('nan'),
        acceptance_radius_m=float('nan'),
        yaw_deg=float('nan'),
        camera_photo_distance_m=float('nan'),
        vehicle_action=MissionItem.VehicleAction.NONE,
    ))
    mission_items.append(MissionItem(
        latitude_deg=47.397825620791885,
        longitude_deg=8.5450092830163271,
        relative_altitude_m=10,
        speed_m_s=2.2352,
        is_fly_through=True,
        gimbal_pitch_deg=float('nan'),
        gimbal_yaw_deg=180,
        camera_action=MissionItem.CameraAction.STOP_PHOTO_INTERVAL,
        loiter_time_s=float('nan'),
        camera_photo_interval_s=float('nan'),
        acceptance_radius_m=float('nan'),
        yaw_deg=float('nan'),
        camera_photo_distance_m=float('nan'),
        vehicle_action=MissionItem.VehicleAction.NONE,
    ))

    mission_plan = MissionPlan(mission_items)



    await drone.mission.set_return_to_launch_after_mission(True)
    await drone.mission.upload_mission(mission_plan)
    print("---Mission plan uploaded---")

    # flight
    print("---Arming Drone---")
    await drone.action.arm()
    await asyncio.sleep(1)
    #cannot sleep > 2 since it has to takeoff within a few seconds of arming
    print("---Taking off---")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    # await drone.camera.set_mode(Mode.PHOTO)
    print('---In flight!---')
    print("---Starting mission---")
    await drone.mission.start_mission()

    async for mission_progress in drone.mission.mission_progress():
        print(f"current waypoint: {mission_progress.current} out of total waypoints: {mission_progress.total}")
        if mission_progress.current == mission_progress.total:
            print("---All waypoints reached!---")
            break

    print("--Waiting for HLOC to analyze photos---")
    hloc_pipeline()
    await asyncio.sleep(10)
    print("--Returning to launch point--")
    await drone.action.return_to_launch()
    # wait for drone to get back to original launch position
    await asyncio.sleep(10)
    # land the drone back at the original spot
    print("---Landing---")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
