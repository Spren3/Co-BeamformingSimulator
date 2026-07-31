'''
    Runs batch of simulations
'''
import sim_mab
# from config import Config

print('Batch sim oracle agent ...')
# seed = 42

max_steps_episode = 100
n_steps = 60000

for num_bss in [7]:
    for num_ant in [4]:
        # cfg = Config(num_bss, num_ant, seed)
        # cfg.max_steps_episode = max_steps_episode
        # cfg.n_steps = n_steps

        # print('*** New run: %s' % cfg)
        # oracle
        sim_mab.run_oracle_agent()

print('done')