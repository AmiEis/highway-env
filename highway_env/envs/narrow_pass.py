from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import *
import numpy as np


class NarrowPass(AbstractEnv):

    def __init__(self, config : dict = None):
        super().__init__(config)

    def _make_road(self):
        net = RoadNetwork()
        net.add_lane("b", "a", StraightLane([40, 0], [0, 0],
                                            line_types=(LineType.STRIPED,LineType.CONTINUOUS_LINE), speed_limit=13.88))
        net.add_lane("a", "b", StraightLane([0, 4], [40, 4],
                                            line_types=(LineType.NONE,LineType.CONTINUOUS_LINE),speed_limit=13.88))
        net.add_lane("b1","b",
                     SineLane([50,2], [40, 0],0.5,np.pi/20,0.5*np.pi,line_types=[LineType.NONE, LineType.CONTINUOUS_LINE],
                              speed_limit=8.33))
        net.add_lane("b","b1",
                     SineLane([40,4],[50,2],0.5,np.pi/20,0.5*np.pi,line_types = [LineType.NONE, LineType.CONTINUOUS_LINE],
                              speed_limit=8.33))
        net.add_lane("b1","c1",StraightLane([50,2], [70, 2], line_types=(LineType.CONTINUOUS_LINE,LineType.CONTINUOUS_LINE),
                                            speed_limit=8.33))
        net.add_lane("c", "c1",
                     SineLane([80, 4], [70, 2], 0.5, np.pi / 20, 0.5 * np.pi,
                              line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
                              speed_limit=8.33))
        net.add_lane("c1", "c",
                     SineLane([70, 2], [80, 0], 0.5, np.pi / 20, -0.5 * np.pi,
                              line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
                              speed_limit=8.33))
        net.add_lane("c","d",StraightLane([80,4],[120,4],
                                          line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
                                          speed_limit=13.88))
        net.add_lane("d", "c", StraightLane([120, 0], [80, 0],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE),
                                            speed_limit=13.88))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        road = self.road
        route = [("a","b",0),("b","b1",0),("b1","c1",0),("c1","c",0),("c","d",0)]
        ego_vehicle = self.action_type.vehicle_class(road,
                                                 road.network.get_lane(("a", "b", 0)).position(20, 0),
                                                 speed=12,
                                                 route=route)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_speed = 12
        other = other_vehicles_type.create_random(self.road, speed=other_speed)
        road.vehicles.append(other)

    def _reset(self):
        self._make_road()
        self._make_vehicles()

    def _reward(self,action):
        return 0

    def _is_terminal(self):
        if self.road.vehicles[0].position[0] > 80:
            return True
        else:
            return False

