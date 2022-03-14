simulation_frequency = 12
policy_frequency = 3
vehicles_density = 1
right_lane_reward = 0.01
high_speed_reward = 0.2
collision_reward = -3.0

duration_train_sec = 10
duration_train_steps = simulation_frequency * duration_train_sec
vehicles_train_count = 10

duration_test_sec = 30
duration_test_steps = simulation_frequency * duration_test_sec
vehicles_test_count = 60

base_config = {"simulation_frequency": simulation_frequency,
               "policy_frequency": policy_frequency,
               "vehicles_density": vehicles_density,
               "high_speed_reward":high_speed_reward,
               "right_lane_reward": right_lane_reward,
               "collision_reward":collision_reward}

train_config = {"duration":duration_train_steps,
                "vehicles_count":vehicles_train_count}

train_config.update(base_config)

test_config = {"duration": duration_test_steps,
               "vehicles_count": vehicles_test_count}

image_obs_config = {
        "observation": {
            #"type": "GrayscaleObservation",
            "type": "MyImageObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    }

def get_config(is_test=False, obs_type="image"):
    config = base_config
    if obs_type == "image":
        config.update(image_obs_config)
    else:
        raise TypeError('obs time currently not supported in configs')
    if is_test:
        config.update(test_config)
    else:
        config.update(train_config)
    return config

if __name__ =="__main__":
    print(get_config())
    print(get_config(is_test=True))






