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

from action_wrapper import NormalizeActions
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

"""figuring out action selection being none"""
    

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
    env = NormalizeActions(env)
    vec_env = make_vec_env(lambda: env)
    norm_env = VecNormalize(vec_env, training=True, norm_obs=True, clip_obs=100, gamma=0.99)
    model = PPO(
        ActorCriticPolicy,
        norm_env,
    )
    sb3_config = config["SB3_config"]
    timesteps = sb3_config.get("timesteps", 1_000_000)
    # asyncio.run(env.compute_state())
    """---SET SB3 LOGGER---"""
    train_path = r"/workspace/sb3_results/"
    if train_path:
        model_logger = configure(train_path, ['csv'])
        model.set_logger(model_logger)
    else:
        model_logger = None
        
    model.learn(total_timesteps=timesteps)

    # asyncio.run(run())
