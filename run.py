# run our best trained agent on random games
from stable_baselines import SAC
from pybulletsim import init_simulation, end_simulation
from env import Drone3DEnv
import time

drone, marker = init_simulation(render = True)
env = Drone3DEnv(drone, marker)
model = SAC.load('models/sac_stablebaseline_1.zip')    # path to the best model


obs = env.reset()
print('initial state: ', obs)

for _ in range(10000):
    action = model.predict(obs)
    obs, rew, done, info = env.step(action[0])
    time.sleep(0.01)
    if done:
        env.reset()

end_simulation()