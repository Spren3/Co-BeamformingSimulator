import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper
from config import Config

# load results from file
res = {}

for num_bss in range(2,4):
    for num_ant in range(2,4):
        config = Config(num_bss, num_ant)
        out_fname = 'data/' + 'rnd_sim_mab_' + config.fname_str() + '.npy'
        (agent_type, agent_cfg, agent_reward) = helper.load_results_from_file(out_fname)
        res['R_B' + str(num_bss) + '_N' + str(num_ant)] = agent_reward

df = pd.DataFrame(data=res)

sns.boxplot(data=df)
plt.ylim([-5, 15])
plt.grid()
plt.ylabel('Reward')
plt.title('Pure random agent')
plt.show()