from gymnasium.core import Env, ObservationWrapper
from waypoint_handler import UpdatedWaypointHandler
from gymnasium import gym
from gymnasium.spaces import Box
import numpy as np


class BSVFlattenWaypointEnv(ObservationWrapper):
    def __init__(self, env: Env, BSV_waypoints: None | np.ndarray = None):
        """__init__.

        Args:
            env (Env): a PyFlyt Waypoints environment.
            context_length: how many waypoints should be included in the flattened observation space
            BSV_waypoints: the waypoints being sampled by BSV to PyFlyt

        """
        super().__init__(env=env)
        if not hasattr(env, "waypoints") and not isinstance(
            env.unwrapped.waypoints,  # type: ignore[reportAttributeAccess]
            UpdatedWaypointHandler,
        ):
            raise AttributeError(
                "Only a waypoints environment can be used with the `FlattenWaypointEnv` wrapper."
            )
        self.BSV_waypoints = BSV_waypoints
        context_length = len(BSV_waypoints) if BSV_waypoints is not None else 0
        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[0]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_shape + self.target_shape * self.context_length,),
        )


    def reset(self, **kwargs):
        """Override the reset method to handle BSV_waypoints."""
        BSV_waypoints = kwargs.pop('BSV_waypoints', None)
        if BSV_waypoints is not None:
            self.BSV_waypoints = BSV_waypoints
        
        # Call the reset method of the wrapped environment, passing BSV_waypoints
        obs, info = self.env.reset(BSV_waypoints=self.BSV_waypoints, **kwargs)
        return self.observation(obs), info

    def observation(self, observation) -> np.ndarray:
        """Flattens an observation from the super env.

        Args:
            observation: a dictionary observation with an "attitude" and "target_deltas" keys.

        """
        num_targets = min(
            self.context_length, observation["target_deltas"].shape[0]
        )  # pyright: ignore[reportGeneralTypeIssues]

        targets = np.zeros((self.context_length, self.target_shape))
        targets[:num_targets] = observation["target_deltas"][
            :num_targets
        ]  # pyright: ignore[reportGeneralTypeIssues]

        new_obs = np.concatenate(
            [observation["attitude"], *targets]
        )  # pyright: ignore[reportGeneralTypeIssues]
        return new_obs