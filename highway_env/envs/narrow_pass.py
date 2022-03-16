from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork


class NarrowPass(AbstractEnv):

    def _make_road(self):
        net = RoadNetwork()
        
        net.add_lane("r_s","r_e")