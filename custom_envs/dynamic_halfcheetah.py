import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
from ray.tune.registry import register_env
from gymnasium.envs.registration import register

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class CustomHalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._current_step = 0
        self._total_step = 0
        self._max_episode_time = 1000
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        self._current_step += 1
        self._total_step += 1
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity

        

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()

        terminated = self._current_step >= self._max_episode_time

        # change the dynamics of reward after ~50M timesteps (after stabilization)
        # could also change a physical attribute, but seems a little more difficult
        # will look into this if the reward weight doesn't do much
        if self._total_step % 1_500_5000 == 0:
            self._forward_reward_weight += 0.4
            print(f"changing forward reward to {self._forward_reward_weight}")
        return observation, reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        return self.reset_model(), {}
        

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self._current_step = 0
        return observation
        
# custom env. needs registered within gymnasium AND also RLLIB (below)
register(
    id='CustomHalfCheetah-v0',
    entry_point=CustomHalfCheetahEnv, 
)

# RLLIB registration
def env_creator(config):
    return CustomHalfCheetahEnv(**config)
register_env("CustomHalfCheetahEnv", env_creator)


