large_obs_dims = (256, 64)
#Use with DQN, TD3, SAC
small_obs_dims = (160, 64)


simulation_frequency = 6
policy_frequency = 2
vehicles_density = 1
right_lane_reward = 0.01
high_speed_reward = 0.1
collision_reward = -3.0
reward_rear_brake = -0.3
reward_off_road = -3.0
reward_front_dist = -0.3
reward_non_centered = -0.03

duration_train_sec = 30
duration_train_steps = policy_frequency * duration_train_sec
vehicles_train_count = 10

duration_test_sec = 90
duration_test_steps = policy_frequency * duration_test_sec
vehicles_test_count = 30


base_config = {"simulation_frequency": simulation_frequency,
               "policy_frequency": policy_frequency,
               "vehicles_density": vehicles_density,
               "high_speed_reward": high_speed_reward,
               "right_lane_reward": right_lane_reward,
               "collision_reward": collision_reward,
               "reward_rear_brake": reward_rear_brake,
               "reward_front_dist": reward_front_dist,
               "reward_off_road": reward_off_road,
               "centering_position": [0.35, 0.5]}

train_config = {"duration": duration_train_steps,
                "vehicles_count": vehicles_train_count}

train_config.update(base_config)

test_config = {"duration": duration_test_steps,
               "vehicles_count": vehicles_test_count}

image_obs_config = {
        "observation": {
            #"type": "GrayscaleObservation",
            "type": "MyImageObservation",
            "observation_shape": large_obs_dims,
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 2.5,
        },
    }

action_config = {
    "action": {
                "type": "DiscreteMetaAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.05,0.05],
                "actions_per_axis": 5,
            },
}


def get_config(is_test=False, obs_type="image"):
    config = base_config
    if obs_type == "image":
        config.update(image_obs_config)
    else:
        raise TypeError('obs type currently not supported in configs')
    config.update(action_config)
    if is_test:
        config.update(test_config)
    else:
        config.update(train_config)
    return config

if __name__ =="__main__":
    print(get_config())
    print(get_config(is_test=True))






