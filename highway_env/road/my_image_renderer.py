import cv2 as cv
import numpy as np
import highway_env.envs.highway_env_scene as highway_env_scene
import matplotlib.pyplot as plt
from highway_env.road.lane import LineType


class MyImageRenderer:
    STRIPES_LENGTH = 2
    STRIPES_SEP = 4

    def __init__(self, env :highway_env_scene, config = None):
        self.config = config or env.config
        self.env = env
        self.image_dims = (self.config["screen_width"], self.config["screen_height"])
        self.image = np.zeros(self.image_dims).astype(np.uint8)
        self.start_stripes = 0
        self.strip_length = int(self.STRIPES_LENGTH * self.config["scaling"])
        self.strip_sep = int(self.STRIPES_SEP * self.config["scaling"])
        self.prev_pos = None

    def center2tl(self, pt):
        centering_position = self.config["centering_position"]
        return np.array([pt[0] + centering_position[0] * self.config["screen_width"],
                         centering_position[1] * self.config["screen_height"] + pt[1]])

    def add_vehicle(self, loc, α, size, color):
        image = self.image
        rot = np.array([[np.cos(α), -np.sin(α)], [np.sin(α), np.cos(α)]])
        loc = np.asarray(loc)
        bottom_left = loc + 0.5 * rot @ ([-size[0], -size[1]])
        bottom_right = loc + 0.5 * rot @ ([size[0], -size[1]])
        top_left = loc + 0.5 * rot @ [-size[0], size[1]]
        top_right = loc + 0.5 * rot @ [size[0], size[1]]
        pts = np.array([bottom_right, bottom_left, top_left, top_right], np.int32)
        image = cv.line(image, pts[0], pts[1], color, 1)
        image = cv.line(image, pts[2], pts[1], color, 1)
        image = cv.line(image, pts[2], pts[3], color, 1)
        image = cv.line(image, pts[0], pts[3], color, 1)
        if color == 2:
            image = cv.line(image, pts[2], pts[0], color, 1)
            image = cv.line(image, pts[1], pts[3], color, 1)
        self.image = image

    def add_lane(self, lat, color, type):
        if type == LineType.STRIPED:
            strip_start = self.start_stripes
            while strip_start < self.image_dims[0]:
                self.image = cv.line(self.image, [lat, strip_start], [lat, strip_start + self.strip_length], color, 1)
                strip_start += self.strip_length + self.strip_sep
        elif type == LineType.CONTINUOUS_LINE:
            self.image = cv.line(self.image, [lat, 0], [lat, self.image_dims[0]], color, 1)

    def my_render(self):
        self.image = np.zeros(self.image_dims).astype(np.uint8)
        scaling = self.config["scaling"]
        agent = self.env.vehicle
        origin = agent.position
        prev_pos = self.prev_pos or origin[0]
        self.start_stripes += int(np.round(((prev_pos - origin[0]) * scaling)))
        self.prev_pos = origin[0]
        while self.start_stripes < - self.strip_length: self.start_stripes += self.strip_length + self.strip_sep
        lane_width = agent.lane.width_at(agent.lane.start[1]) * scaling
        road = self.env.road
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for ind, l in enumerate(road.network.graph[_from][_to]):
                    agent_in_lane_cs = -np.array(l.local_coordinates(np.array(origin)))*scaling
                    lane_in_im = self.center2tl(agent_in_lane_cs)
                    lane_lat = int(lane_in_im[1] - 0.5*lane_width)
                    self.add_lane(lane_lat, 1, l.line_types[0])
                    self.add_lane(lane_lat + int(lane_width), 1, l.line_types[1])
        color = 3
        for vehicle in road.vehicles:
            v_size = (vehicle.LENGTH * scaling, vehicle.WIDTH * scaling)
            v_pos = self.center2tl(scaling * (vehicle.position - origin))
            v_head = -vehicle.heading
            self.add_vehicle(np.array([v_pos[1], v_pos[0]]), v_head, (v_size[1], v_size[0]), color)
            color = 2
        if not self.config["offscreen_rendering"]:
            return self.image.swapaxes(0,1)
            #plt.imshow(self.image.swapaxes(0,1))

    def reset_pos(self):
        self.prev_pos = None