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
from drone_hover_env import DroneHoverEnv
from mavsdk.camera import Mode, Setting, Option
from mavsdk.mission import MissionItem, MissionPlan
from stable_baselines3.common.policies import ActorCriticPolicy
from mavsdk.offboard import OffboardError, PositionNedYaw, Attitude
    

def load_yaml_config(file_path):
    """Load the YAML configuration file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    yaml_config_path = r"/workspace/config_file.yaml"
    config = load_yaml_config(yaml_config_path)

    hover_config = config['hover_env_config']
    env = DroneHoverEnv(hover_config)
    model = PPO(
        ActorCriticPolicy,
        env,
    )
    sb3_config = config["SB3_config"]
    timesteps = sb3_config.get("timesteps", 1_000_000)
    # asyncio.run(env.compute_state())
    model.learn(total_timesteps=timesteps)

    # asyncio.run(run())
