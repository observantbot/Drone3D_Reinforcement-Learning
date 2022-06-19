from pybulletsim import init_simulation, end_simulation
from env import Drone3DEnv
from stable_baselines import SAC
import pandas as pd

drone, marker = init_simulation(render = False)
env = Drone3DEnv(drone, marker)


def run_agent(model):
    obs = env.reset()
    print('initial state: ', obs)
    t = 0
    for i in range(500):
        action = model.predict(obs)
        obs, reward, done, _ = env.step(action[0])
        # time.sleep(0.01)
        t += 0.01
        if done:
            if reward>4.5:
                print('***********',t)
            else:
                print('***********', 5)     # evaluation metric
                t = 5
            return t
    return t

t1 = []
game = []
for i in range(1,51):
    filepath = 'models/sac_stablebaseline_'+ str(i) + '.zip'
    model = SAC.load(filepath)
    t = run_agent(model=model)
    t1.append(t)
    game.append(i)


d = {'game': game, 'time': t1}
df = pd.DataFrame(d).to_csv('csv/game.csv')


end_simulation()