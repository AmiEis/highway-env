from stable_baselines3 import PPO
import gym
import highway_env



if __name__ == "__main__":
    model = PPO.load("highway_ppo/model")
    env = gym.make("highway-fast-v0")
    env.seed(1234)
    env.reset()
    n_collisions = 0
    n_off_road = 0
    n_success = 0
    n_scenes = 100
    for _ in range(n_scenes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, reward, done, info = env.step(action)
            if env.vehicle.crashed:
                n_collisions += 1
            if env.steps >= env.config["duration"] and not env.vehicle.crashed:
                n_success += 1
            if env.config["offroad_terminal"] and not env.vehicle.on_road:
                n_off_road += 1
            #env.render()
    print("Sucess rate = ",float(n_success)/n_scenes)
    print("Num collisions: {} out of {} scenes".format(n_collisions,n_scenes))