from ctypes import *
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle

DOUBLE2 = c_double * 2


class Vehicle(Structure):
    _fields_ = [("speed", c_double), ("target_speed", c_double), ("position", DOUBLE2),
                ("velocity", DOUBLE2), ("heading", c_double),("lane_index", c_int),
                ("target_lane_index", c_int), ("timer", c_double), ("DELTA", c_double),
                ("Length", c_double), ("action_steering", c_double), ("action_acceleration", c_double)]


def vehicle_py_to_ctypes(v_py):
    v_ctypes = Vehicle()
    v_ctypes.speed = v_py.speed
    v_ctypes.target_speed = v_py.target_speed
    v_ctypes.position = DOUBLE2(v_py.position[0], v_py.position[1])
    v_ctypes.velocity = DOUBLE2(v_py.velocity[0], v_py.velocity[1])
    v_ctypes.heading = v_py.heading
    v_ctypes.lane_index = v_py.lane_index[2]
    v_ctypes.target_lane_index = v_py.target_lane_index[2]
    if isinstance(v_py,IDMVehicle):
        v_ctypes.timer = v_py.timer
        v_ctypes.DELTA = v_py.DELTA
    v_ctypes.Length = v_py.LENGTH
    v_ctypes.action_steering = v_py.action["steering"]
    v_ctypes.action_acceleration = v_py.action["acceleration"]
    return v_ctypes


class ActAccelerator:
    def __init__(self):
        accelerator_dll = CDLL(r'D:\projects\RL\highway-env-accel\cmake-build-release-visual-studio\Release\he-accel.dll')
        init_accelerator = accelerator_dll.init_accelerator
        init_accelerator.restype = c_void_p
        self.accelerator = init_accelerator()
        act_all = accelerator_dll.act_all
        act_all.argtypes = [c_void_p, POINTER(Vehicle), c_int]
        self.act_all = act_all

    def act_others(self,vehicles):
        ctypes_vehicles_py = []
        for v in vehicles:
            ctypes_vehicles_py.append(vehicle_py_to_ctypes(v))
        sz = len(vehicles)
        ctypes_vehicles = (Vehicle * sz)(*ctypes_vehicles_py)
        self.act_all(self.accelerator,ctypes_vehicles,c_int(sz))
        for v_py, v_c in zip(vehicles[1:], list(ctypes_vehicles[1:sz])):
            v_py.update_py_vehicle_from_ctypes(v_c)


if __name__ == "__main__":
    test_accelerator = True
    test_translation = False
    road = Road(network=RoadNetwork.straight_road_network(1))
    if test_accelerator:
        actAccelerator = ActAccelerator()
        py_vehicles = []
        for i in range(2):
            py_vehicles.append(IDMVehicle.create_random(road))
        print(py_vehicles[0].speed)
        print(py_vehicles[1].speed)
        actAccelerator.act_others(py_vehicles)
        print(py_vehicles[0].speed)
        print(py_vehicles[1])



    if test_translation:
        py_vehicle = IDMVehicle.create_random(road)
        ctypes_res = vehicle_py_to_ctypes(py_vehicle)
        print('Test python to ctypes conversion;')
        for name,c_type in ctypes_res._fields_:
            val = getattr(ctypes_res, name)
            if c_type == DOUBLE2:
                print(name, val[0], val[1])
            else:
                print (name, val)
        py_vehicle_2 = IDMVehicle.create_random(road)
        print('Test ctypes to python conversion:')
        print('Original python vehicle')
        for att in vars(py_vehicle_2):
            print(att, getattr(py_vehicle_2,att))
        py_vehicle_2.update_py_vehicle_from_ctypes(ctypes_res)
        print('Pyhton vehicle after update')
        for att in vars(py_vehicle_2):
            print(att, getattr(py_vehicle_2,att))
