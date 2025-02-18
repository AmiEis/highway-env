import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.common.observation import MyHighwayGrayscale
from highway_env.envs.common.cpp_accelerator import ActAccelerator


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    def __init__(self, config: dict = None, use_safety_control=False):
        super().__init__(config)
        self.use_safety_control = use_safety_control
        self.actAccelerator = ActAccelerator()
        self.lat_speed_buffer = np.zeros(self.config["lat_speed_buffer_size"])

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "target_speed": 30,
            "reward_rear_brake": -0.4,
            "reward_rear_deceleration_range": [3.5, 6],
            "reward_front_dist": -0.4,
            "reward_front_dist_range": [10, 20],
            "reward_non_centered": -0.05,
            "reward_off_road": -1,
            "offroad_terminal": False,
            "speed_limit": 30,
            "lat_speed_buffer_size": 6
        })
        config["reward_speed_range_lower"] = [config["target_speed"] - 10,config["target_speed"]]
        config["reward_speed_range_upper"] = [config["target_speed"],config["target_speed"] + 5]
        return config


    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=self.config["speed_limit"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            #Added so that the ego will not necessarily be at the back position
            #It will be randomly positioned between the other vehicles
            split = self.np_random.randint(others//2)
            for _ in range(split):
                target_speed = self.np_random.uniform(1.0*self.config["speed_limit"],1.2*self.config["speed_limit"])
                speed = target_speed + self.np_random.normal(0,2)
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"],
                                                            speed=speed)
                vehicle.target_speed = target_speed
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            #self.road.vehicles.append(controlled_vehicle)
            self.road.vehicles.insert(0,controlled_vehicle)
            for _ in range(split,others):
                target_speed = self.np_random.uniform(0.8*self.config["speed_limit"],0.95*self.config["speed_limit"])
                speed = target_speed + self.np_random.normal(0,2)
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"],
                                                            speed=speed)
                vehicle.target_speed = target_speed
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        if self.vehicle.speed <= self.config["target_speed"]:
            scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range_lower"], [0, 1])
        else:
            scaled_speed = 1 - utils.lmap(self.vehicle.speed, self.config["reward_speed_range_upper"], [0, 1])
        rear_break, front_dist = self.calc_rear_break_front_dist()
        scaled_deceleration = utils.lmap(-rear_break,self.config["reward_rear_deceleration_range"], [0,1])
        scaled_front_distance = 1 - utils.lmap(front_dist, self.config["reward_front_dist_range"],[0,1])
        dist_from_lane_center = abs(self.vehicle.lane.local_coordinates(self.vehicle.position)[1])
        scaled_dist_from_lane_center = utils.lmap(dist_from_lane_center,[0,0.5*self.vehicle.lane.DEFAULT_WIDTH],[0,1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["reward_rear_brake"] * np.clip(scaled_deceleration, 0, 1)\
            + self.config["reward_front_dist"] * np.clip(scaled_front_distance, 0, 1)
            #+ self.config["reward_off_road"] * (not self.vehicle.on_road)
        #reward = utils.lmap(reward,
        #                  [self.config["collision_reward"],
        #                   self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                  [0, 1])
        if not isinstance(self.action_type,DiscreteMetaAction):
            reward += self.config["reward_non_centered"] * np.clip(scaled_dist_from_lane_center, 0, 1)
        reward = -self.config["high_speed_reward"] if not self.vehicle.on_road else reward
        return reward

    def calc_rear_break_front_dist(self, is_test=False):
        #calculated in one go so the two calculations have to find front/rear
        # and thus iterate on all vehicles
        front, rear = self.road.neighbour_vehicles(self.vehicle)
        if not rear is None:
            #Find if agent is changing lane, and if its follower is forced to make an emergency break
            rear_acc = rear.action['acceleration']
            #check that the agent is changing lane, and the following vehicle is in the target lane
            is_lane_change = abs(self.vehicle.heading) > np.deg2rad(5) \
                            and self.vehicle.velocity[1] * (rear.position[1] - self.vehicle.position[1]) > 0
            if isinstance(self.observation_type,MyHighwayGrayscale):
                is_in_obs = self.observation_type.renderer.is_in_image(rear)
                is_lane_change = is_lane_change and is_in_obs # Cannot punish aggressive behaviour when vehicle not in sight
        else:
            rear_acc = 0
            is_lane_change = False
        if front is not None:
            dist = front.position[0] - self.vehicle.position[0]
        else: dist = np.inf
        if is_test:
            return is_lane_change, rear_acc, dist
        return is_lane_change * rear_acc, dist

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
