# ensure custom environment matches gym env interface 
from stable_baselines.common.env_checker import check_env
from pybulletsim import init_simulation, end_simulation
from env import Drone3DEnv


drone, marker = init_simulation()
env = Drone3DEnv(drone, marker)
print('obs_space',env.observation_space)
print('action_spc: ', env.action_space)

for i in range(10):
    print(env.action_space.sample())

check_env(env, warn=True)
end_simulation()