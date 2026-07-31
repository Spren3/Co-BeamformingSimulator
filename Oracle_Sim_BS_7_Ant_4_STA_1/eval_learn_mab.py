import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper

# load from file
with open('data/rnd_sim_mab.npy', 'rb') as f:
    rnd_type = np.load(f)
    rnd_cfg = np.load(f)
    rnd_reward = np.load(f)

# load from file
with open('data/learn2_sim_mab.npy', 'rb') as f:
    learn_type = np.load(f)
    learn_cfg = np.load(f)
    learn_reward = np.load(f)


# load from file
with open('data/ora2_sim_mab.npy', 'rb') as f:
    ora_type = np.load(f)
    ora_cfg = np.load(f)
    ora_reward = np.load(f)

# check if both are same cfg
#assert np.array_equal(rnd_cfg, learn_cfg)


last_n_steps = 1000
d = {'Random': rnd_reward[-last_n_steps:], 'NN-CMAB': learn_reward[-last_n_steps:], 'Oracle': ora_reward[-last_n_steps:]}
df = pd.DataFrame(data=d)

mv = [np.mean(d[k]) for k in d]
print(mv)
gain = (mv - mv[0]) / mv[0] * 100
print(gain)

sns.boxplot(data=df)
plt.ylim([-5, 15])
plt.grid()
plt.ylabel('Reward')
plt.title('Scenario: 2 OBSS, Ant=2')
plt.show()
