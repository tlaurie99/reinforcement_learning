from waypoints_flat_wrapper import BSVFlattenWaypointEnv
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import gymnasium as gym
import argparse
import time
from waypoints_env import Quadx_waypoints_2


def register_the_env():
    register(
        id='PyFlyt/QuadXBSV-Waypoints-v1', 
        entry_point=Quadx_waypoints_2,
        max_episode_steps=500, 
    )

def parse_args():
    parser = argparse.ArgumentParser(description="PyFlyt Waypoint Evaluation")
    parser.add_argument('--waypoints', nargs=3, type=float, required=True, help="Waypoint coordinates (x, y, z)")
    args = parser.parse_args()
    return np.array(args.waypoints).reshape(1, 3)


def evaluate(model_path, waypoints_input):
    init_waypoints = [[0, 1, 1]]
    register_the_env()
    env = BSVFlattenWaypointEnv(gym.make(id='PyFlyt/QuadXBSV-Waypoints-v1', flight_mode=-1), BSV_waypoints=init_waypoints)

    env.action_space = spaces.Box(low = np.array(
             [
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
             ]
         ),  high = np.array(
             [
                 1.0,
                 1.0,
                 1.0,
                 1.0,
             ]
         ), dtype=np.float64)

    
    model_loaded = PPO.load(model_path, env=env)
    # f'/mnt/c/Users/tyler/OneDrive/Desktop/pyflyt/best_model.zip'


    obs_list = []
    obs, info = env.reset(BSV_waypoints=waypoints_input)

    reward_list = []
    action_list = []
    target_list = []
    obs_array_list = []
    start = time.time()
    terminated = False
    while not terminated:
        action, _states = model_loaded.predict(obs,
                                        deterministic=True
                                        )
    # obs, reward, terminated, truncated, info = env.step(np.zeros((4))+.79)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"info is: {info}")

        obs_list += [obs]
        reward_list += [reward]
        action_list += [action]

        # if info['num_targets_reached'] < len(waypoints_input):
        #     print("failure")
        # else:
        #     print("success")


        if terminated:
            print("failure")
            break
        elif info['num_targets_reached'] == 1:
            print("success")
            break

    env.close()

    obs_array = np.array(obs_list)
    reward_array = np.array(reward_list)
    print(f"reward: {reward_array}")
    action_array = np.array(action_list)
    # targets_array = np.array(target_waypoint_local)

if __name__ == "__main__":
    waypoints = parse_args()
    evaluate('src/best_model.zip', waypoints)