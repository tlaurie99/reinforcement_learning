import os
import yaml
import numpy as np
import aioitertools
import torch.nn as nn
from stable_baselines3 import PPO
from drone_hover_env import DroneHoverEnv
from action_wrapper import NormalizeActions
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

"""Main method for running PX4 / MAVSDK / SB3 training pipeline"""

def load_yaml_config(file_path):
    """Load the YAML configuration file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def separate_configs(config):
    hover_config = config['hover_env_config']
    sb3_config = config['SB3_config']
    return hover_config, sb3_config

def get_sb3_params(sb3_config):
    learning_rate = sb3_config.get('learning_rate', 0.0003)
    n_steps = sb3_config.get('n_steps', 2048)
    batch_size = sb3_config.get('batch_size', 64)
    n_epochs = sb3_config.get('n_epochs', 10)
    timesteps = sb3_config.get("timesteps", 1_000_000)
    train_path = sb3_config.get("train_path", '/sb3_results/')

    policy_kwargs = {
        'activation_fn': nn.LeakyReLU,
        'net_arch': [64, 64]
    }
    return learning_rate, n_steps, batch_size, n_epochs, timesteps, train_path, policy_kwargs

if __name__ == "__main__":
    yaml_config_path = r"/workspace/config_file.yaml"
    config = load_yaml_config(yaml_config_path)
    # set the environment
    hover_config, sb3_config = separate_configs(config)
    env = DroneHoverEnv(hover_config)
    env = NormalizeActions(env)
    vec_env = make_vec_env(lambda: env)
    norm_env = VecNormalize(vec_env, training=True, norm_obs=True, clip_obs=100, gamma=0.99)
    # get SB3 params
    lr, steps, batch, epochs, timesteps, path, policy_kwargs = get_sb3_params(sb3_config)
    # build the model
    model = PPO(
        ActorCriticPolicy,
        norm_env,
        learning_rate=lr,
        n_steps=steps,
        batch_size=batch,
        n_epochs=epochs,
        verbose=1,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
    )
    
    """---SET SB3 LOGGER---"""
    train_path = path
    if train_path:
        model_logger = configure(train_path, ['csv'])
        model.set_logger(model_logger)
    else:
        model_logger = None
        
    model.learn(total_timesteps=timesteps)
