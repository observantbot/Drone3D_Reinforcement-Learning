# train a DDPG algorithm based model using Stable Baselines library from open AI.

import numpy as np
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from pybulletsim import init_simulation, end_simulation
from env import Drone3DEnv


drone, marker = init_simulation()
env = Drone3DEnv(drone, marker)

model = SAC(MlpPolicy, env, verbose=1)

for i in range(1, 51):
  save_path = "models/sac_stablebaseline_" + str(i) + ".zip"
  if i==1:
    model.learn(total_timesteps=10000)
    # at every 10000 time steps, we will save our model
    model.save(save_path)
  else:
    del model
    model = SAC.load(prev_path, env)
    model.learn(total_timesteps=10000)
    model.save(save_path)
  prev_path = save_path

print('done')

end_simulation()