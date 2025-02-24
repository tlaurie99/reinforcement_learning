import PyFlyt.gym_envs
from pettingzoo import AECEnv
from ray.tune.registry import register_env
from PyFlyt.gym_envs import FlattenWaypointEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from PyFlyt.gym_envs.quadx_envs import quadx_hover_env, quadx_waypoints_env
from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_dogfight_env import MAFixedwingDogfightEnv

'''
This is a custom PyFlyt Env. based on the MAFixedwingDogfightEnv which allows for vectorization
'''
class CustomDogfightEnv(MultiAgentEnv):
    def __init__(self, 
                 config, 
                 env: AECEnv = None):

        super().__init__()
        if env is None:
            self.env = MAFixedwingDogfightEnv()
        else:
            self.env = env
        self.env.reset()
        
        self.agent_ids = self.env.possible_agents
        self.observation_space = self.env.observation_space(self.env.agents[0])
        self.action_space = self.env.action_space(self.env.agents[0])

        # self.custom_reward_wrapper = CustomRewardWrapper(self.env)

        assert all(
            self.env.observation_space(agent) == self.observation_space
            for agent in self.env.agents
        ), (
            "Observation spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_observations wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_observations(env)`"
        )

        assert all(
            self.env.action_space(agent) == self.action_space
            for agent in self.env.agents
        ), (
            "Action spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_action_space wrapper can help (usage: "
            "`supersuit.aec_wrappers.pad_action_space(env)`"
        )
        self._agent_ids = set(self.env.agents)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset()
        return observations, infos

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        # ensure "__all__" keys are present in terminations and truncations dictionaries
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values())
        
        return observations, rewards, terminations, truncations, infos

'''
Register the env within RLLIB
'''
def env_creator(config):
    return CustomDogfightEnv(config)
register_env('MAFixedwingDogfightEnv', env_creator)

'''
Policy mapping function to use within RLLIB specific to PyFlyt
'''
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # check if agent is a number
    if agent_id.isdigit():
        return 'policy_1' if int(agent_id) % 2 == 0 else 'policy_2'
    # handles agent_ids like 'uav_0', 'uav_1', etc.
    return 'policy_1' if int(agent_id.split('_')[1]) % 2 == 0 else 'policy_2'
