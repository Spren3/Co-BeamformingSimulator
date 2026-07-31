'''
    Runs batch of simulations
'''
import sim_mab
from config import Config

print('Batch sim random agent ...')
seed = 42

max_steps_episode = 10
n_steps = 1000

for num_bss in [2, 3, 7]:
    for num_ant in [2, 3, 6, 10]:
        cfg = Config(num_bss, num_ant, seed)
        cfg.max_steps_episode = max_steps_episode
        cfg.n_steps = n_steps

        print('*** New run: %s' % cfg)
        # random
        sim_mab.run_random_agent(cfg)

print('done')