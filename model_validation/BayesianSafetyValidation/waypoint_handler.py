from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler
from pybullet_utils import bullet_client
import numpy as np
import math




class UpdatedWaypointHandler(WaypointHandler):
    def __init__(
        self,
        enable_render: bool,
        num_targets: int,
        use_yaw_targets: bool,
        goal_reach_distance: float,
        goal_reach_angle: float,
        flight_dome_size: float,
        min_height: float,
        np_random: np.random.Generator,
        BSV_waypoints: None | np.ndarray = None,
    ):

        super().__init__(
        enable_render=enable_render,
        num_targets=num_targets,
        use_yaw_targets=use_yaw_targets,
        goal_reach_distance=goal_reach_distance,
        goal_reach_angle=goal_reach_angle,
        flight_dome_size=flight_dome_size,
        min_height=min_height,
        np_random=np_random
        )

        self.enable_render = enable_render
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.min_height = min_height
        self.np_random = np_random

    def reset(self, 
              p: bullet_client.BulletClient, 
              np_random: None | np.random.Generator = None, 
              BSV_waypoints: None | np.ndarray = None):
              # add waypoints from BSV here
        # reset the error
        self.p = p
        self.new_distance = np.inf
        self.old_distance = np.inf
        
        if np_random is None:
            self.np_random = np.random.default_rng()
        else:
            self.np_random = np_random

        self.targets = np.zeros(shape=(self.num_targets, 3))
        
        if BSV_waypoints is not None:
            # check to see if BSV passed enough waypoints
            if len(BSV_waypoints) != self.num_targets:
                raise ValueError(f"expected {self.num_targets} BSV_waypoints, but got {len(BSV_waypoints)}")
        
            # continue with adding waypoints/targets to the instance
            '''we don"t have to do the polar -> cartesian transformation since we are allowing BSV to give cartesian coords'''
            for i, waypoint in enumerate(BSV_waypoints):
                x, y, z = waypoint
                # check to see if z is above the floor
                self.targets[i] = [x, y, z if z > self.min_height else self.min_height]
        else:
            # we sample from polar coordinates to generate linear targets
            thetas = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
            phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
            for i, theta, phi in zip(range(self.num_targets), thetas, phis):
                dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
                x = dist * math.sin(phi) * math.cos(theta)
                y = dist * math.sin(phi) * math.sin(theta)
                z = abs(dist * math.cos(phi))
    
                # check for floor of z
                self.targets[i] = np.array(
                    [x, y, z if z > self.min_height else self.min_height]
                )

        print(f"targets are now: {self.targets}")
        # yaw targets
        if self.use_yaw_targets:
            self.yaw_targets = self.np_random.uniform(
                low=-math.pi, high=math.pi, size=(self.num_targets,)
            )

        # if we are rendering, load in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    self.p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )



    