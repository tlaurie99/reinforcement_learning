from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from reinforcement_learning.models import ( 
    ClampedCritic, MOGCritic, ParameterizedCritic, CentralizedCritic, ENNCritic
)

"""
With ClampedCritic and --lr=0.0003
(trying to reach 500.0 return on HalfCheetah in 2M env steps):
+-----------------------------+------------+-----------------+--------+
| Trial name                  | status     | loc             |   iter |
|                             |            |                 |        |
|-----------------------------+------------+-----------------+--------+
| HalfCheetah-v5              | TERMINATED | 127.0.0.1:8888  |    802 |
+-----------------------------+------------+-----------------+--------+
+------------------+------------------------+---------------------+
|   total time (s) | num_env_steps_sampled_ | episode_return_mean |
|                  |              _lifetime |                     |
|------------------+------------------------+---------------------+
|          ------- |                3212000 |              504.1  |
+------------------+------------------------+---------------------+


With MOGCritic / --lr=0.0003 / --num_gaussians=3
(trying to reach 500.0 return on HalfCheetah in 2M env steps):
+-----------------------------+------------+-----------------+--------+
| Trial name                  | status     | loc             |   iter |
|                             |            |                 |        |
|-----------------------------+------------+-----------------+--------+
| HalfCheetah-v5              | TERMINATED | 127.0.0.1:8888  |    645 |
+-----------------------------+------------+-----------------+--------+
+------------------+------------------------+---------------------+
|   total time (s) | num_env_steps_sampled_ | episode_return_mean |
|                  |              _lifetime |                     |
|------------------+------------------------+---------------------+
|          ------- |                2584000 |              500.7  |
+------------------+------------------------+---------------------+


With ParameterizedCritic and --lr=0.0003
(trying to reach 500.0 return on HalfCheetah in 2M env steps):
+-----------------------------+------------+-----------------+--------+
| Trial name                  | status     | loc             |   iter |
|                             |            |                 |        |
|-----------------------------+------------+-----------------+--------+
| HalfCheetah-v5              | TERMINATED | 127.0.0.1:8888  |    628 |
+-----------------------------+------------+-----------------+--------+
+------------------+------------------------+---------------------+
|   total time (s) | num_env_steps_sampled_ | episode_return_mean |
|                  |              _lifetime |                     |
|------------------+------------------------+---------------------+
|          ------- |                2516000 |              505.3  |
+------------------+------------------------+---------------------+
"""

torch, _ = try_import_torch()


parser = add_rllib_example_script_args(
    default_reward=500.0,
    default_timesteps=10_000_000,
)
parser.set_defaults(enable_new_api_stack=False)
parser.add_argument(
    "--environment",
    type=str,
    default="HalfCheetah-v5",
    help="Environment to run from the gymnasium environments",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,
    help="Learning rate",
)

parser.add_argument(
    "--custom_model_type",
    type=str,
    choices=['MOGCritic', 'ClampedCritic', 'ParameterizedCritic'],
    default='MOGCritic',
    help="Custom critic component type"
)

parser.add_argument(
    "--vf_share_layers",
    type=bool,
    default=False,
    help="Whether to share layers with actor network"
)

parser.add_argument(
    "--fcnet_hiddens",
    type=list,
    default=[256, 256],
    help="Whether to share layers with actor network"
)

parser.add_argument(
    "--fcnet_activation",
    type=str,
    default="LeakyReLU",
    help="Activation function type"
)

if __name__ == "__main__":
    args = parser.parse_args()

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(args.environment)
        .training(
            lr=args.lr,
            model={'custom_model': args.custom_model_type, 'vf_share_layers': args.vf_share_layers, 
           'fcnet_hiddens': args.fcnet_hiddens,'fcnet_activation': args.fcnet_activation},
        )
    )

algo = algo_config.build()
total_timesteps = 0
default_reward = 500.0
default_timesteps = 10_000_000
for iteration in range(1000):
    result = algo.train()



    episode_return_mean = result["env_runners"]["episode_return_mean"]
    total_timesteps = result["num_env_steps_sampled_lifetime"]

    pprint(
        f"Iteration {iteration}: Episode Return Mean: {episode_return_mean}, "
        f"Total Timesteps: {total_timesteps}"
    )

    if episode_return_mean >= default_reward:
        print(f"Stopping: Reached target reward of {default_reward} at iteration {iteration}")
        break
    if total_timesteps >= default_timesteps:
        print(f"Stopping: Reached maximum timesteps of {default_timesteps} at iteration {iteration}")
        break