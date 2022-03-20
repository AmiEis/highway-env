from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import *
import numpy as np


class NarrowPass(AbstractEnv):

    def __init__(self, config : dict = None):
        super().__init__(config)

    def _make_road(self):
        net = RoadNetwork()
        net.add_lane("a", "b", StraightLane([0, 0], [40, 0],
                                            line_types=(LineType.CONTINUOUS_LINE,LineType.STRIPED)))
        net.add_lane("a", "b", StraightLane([0, 4], [40, 4],
                                            line_types=(LineType.STRIPED,LineType.CONTINUOUS_LINE)))
        net.add_lane("b","b1",
                     SineLane([40,0],[50,2],0.5,np.pi/20,-0.5*np.pi,line_types=[LineType.CONTINUOUS_LINE, LineType.NONE]))
        net.add_lane("b","b1",
                     SineLane([40,4],[50,2],0.5,np.pi/20,0.5*np.pi,line_types = [LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("b1","c1",StraightLane([50,2], [70, 2]))
        net.add_lane("c1", "c",
                     SineLane([70, 2], [80, 4], 0.5, np.pi / 20, 0.5 * np.pi,
                              line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("c1", "c",
                     SineLane([70, 2], [80, 0], 0.5, np.pi / 20, -0.5 * np.pi,
                              line_types=[LineType.CONTINUOUS_LINE, LineType.NONE]))
        net.add_lane("c","d",StraightLane([80,4],[120,4]))
        net.add_lane("c", "d", StraightLane([80, 0], [120, 0]))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                 road.network.get_lane(("a", "b", 1)).position(20, 0),
                                                 speed=12)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

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

