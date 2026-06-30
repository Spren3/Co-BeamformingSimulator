import random
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import torch.optim as optim
import torch.nn as nn
# from Sim import Sim
from simulator import Sim, Config
# from config import Config


import torch
import copy
import csv
 


"""
-> Deep Deterministic Policy Gradient(DDPG) is the model-free, off-policy deepreinforcement algorithm

-> Basic parameters needed for the Deep Deterministic Policy Gradient Actor Critic Nodel
"""
capacity=1000000
batch_size=512
update_iteration=10
tau=0.001
gamma=0.99
directory = './'
hidden1=32
hidden2=32
MIN_BUFFER_SIZE = 5000


"""
Check if our PC has GPU or not
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("This code will run on ------> GPU")
else:
    print("This code will run on ------> CPU")

print("\n\n")


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.

        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        # print("I am noiseeeeeeeeeeeeeeeeeeee")
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
    

class Replay_buffer():
    """
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    """
    def __init__(self, max_size=capacity):
        """Create Replay buffer.
        : Parameters
        size: int
                : Max number of transitions to store in the buffer. When the buffer
                overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size,it):
        """
        Sample a batch of experiences.
        
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        
        Returns
        -------
        state: np.array
            batch of state or observations
        action: np.array
            batch of actions executed given a state
        reward: np.array
            rewards received as results of executing action
        next_state: np.array
            next state next state or observations seen after executing action
        done: np.array
            done[i] = 1 if executing ation[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # print(len(self.storage),'   ',it)
        # print(batch_size)

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in ind:
            st, n_st, act, rew, dn = self.storage[i]
            state.append(np.asarray(st))
            next_state.append(np.asarray(n_st))
            action.append(np.asarray(act))
            reward.append(np.asarray(rew))
            done.append(np.asarray(dn))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):

    """
    : The Actor model will take the state observation as input and outputs a continuous action value.
    : It consists of four fully coonected linear layers/ you can change accorind to your choice, with ReLU activation functions and 
    a final Sigmoid output layer selects number of optimized action for the state.
    : I used Sigmoid just because our actions are needed in the range of 0 ~ 1 (These will be angles which needs to be null 
    to improve the datarate for our simulation.)
    """

    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # [-1,1]
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    """
    : The Critic model takes in both a state observation and an action as input and outputs a Q-value, which estimates 
    the expected total reward for the current state-action pair. 
    
    : It consists of four linear layers with ReLU activation functions, State and action inputs are concatenated before 
    being fed into the first linear layer. 
    
    : The output layer has a single output, representing the Q-value
    """
    def __init__(self, n_states, action_dim, hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        
    def forward(self, state, action):
        aa = torch.cat((state, action), 1)
        temp = self.net(aa)
        return temp



class DDPG(object):
    def __init__(self, state_dim, action_dim):
        """
        Initializes the DDPG agent. 
        Takes three arguments:
            : state_dim  ----> which is the dimensionality of the state space, 
            : action_dim ----> which is the dimensionality of the action space, and 
            : max_action ----> which is the maximum value an action can take. 
        
        Creates a replay buffer, an actor-critic  networks and their corresponding target networks. 
        It also initializes the optimizer for both actor and critic networks alog with 
        counters to track the number of training iterations.
        """
        self.replay_buffer = Replay_buffer()
        

        self.actor = Actor(state_dim, action_dim, hidden1).to(device)
        self.actor_target = Actor(state_dim, action_dim,  hidden1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=1e-4)


        self.critic = Critic(state_dim, action_dim,  hidden2).to(device)
        self.critic_target = Critic(state_dim, action_dim,  hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)


        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, ou_noise):
        """
        : Takes the current state as input and returns an action to take in that state. 
        : It uses the actor network to map the state to an action.
        """
        
        noise = ou_noise()
        zzz = state.reshape(1, -1)
        state = torch.FloatTensor(zzz).to(device)
        aaa = self.actor(state)             # [-1, 1]
        noise = torch.FloatTensor(noise).to(device)
        ddd = aaa + noise                   # still ~[-1,1]
        scaled = (ddd + 1.0) / 2.0          # → [0,1]
        legal_action = torch.clamp(scaled, 0.0, 1.0)
        
        return legal_action.cpu().data.numpy().flatten()


    def update(self):
        """
        updates the actor and critic networks using a batch of samples from the replay buffer. 
        For each sample in the batch, it computes the target Q value using the target critic network and the target actor network. 
        It then computes the current Q value 
        using the critic network and the action taken by the actor network. 
        
        It computes the critic loss as the mean squared error between the target Q value and the current Q value, and 
        updates the critic network using gradient descent. 
        
        It then computes the actor loss as the negative mean Q value using the critic network and the actor network, and 
        updates the actor network using gradient ascent. 
        
        Finally, it updates the target networks using 
        soft updates, where a small fraction of the actor and critic network weights are transferred to their target counterparts. 
        This process is repeated for a fixed number of iterations.
        """

        for it in range(update_iteration):
            # For each Sample in replay buffer batch
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size,it)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1-done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Compute the target Q value
            # print("next_state---------------->",next_state)
            # print("next_state type---------------->",type(next_state))
            temp = self.actor_target(next_state)
            # print("temp---------------->",temp)
            # print("temp type---------------->",type(temp))
            # print(self.critic_target)
            target_Q = self.critic_target(next_state, temp)
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()
            

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            
            """
            Update the frozen target models using 
            soft updates, where 
            tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
            """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        

    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))








"""
Iterate over different base station with different combinations of antennas
"""

import time
code_start_time = time.time()


if __name__ == '__main__':
    seed = 42
    num_bss = 7
    num_antennas = 2
    config = Config(num_bss, num_antennas, seed)

    np.random.seed(config.seed)
    random.seed(config.seed)
    env = Sim(config)


    action_dim, state_dim = env.get_spaces()
    
    agent = DDPG(state_dim, action_dim)
    std_dev = 0.05
    # ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    ou_noise = OUActionNoise(
        mean=np.zeros(action_dim),
        std_deviation=float(std_dev) * np.ones(action_dim)
    )

    rewards = []
    flag1 = True
    flag2 = True
    allRewards = []

    max_episode = 20000
    allRewards = []

    for i in range(max_episode):
        print(f"Episode {i} | Main_DDPG_Interf_Opt_{num_bss}APs_fastRunning")
        print(f"BS: {config.num_bss} | Antennas: {config.num_antennas} | (STA)s: {config.max_num_stas}")
        obs = env.reset()
        state = obs.flatten()
        # print(state)


        ou_noise.reset()  # reset noise at each episode start

        episode_reward = 0
        t = 0

        while True:

            # Agent action: shape (action_dim,) = (num_bs,)
            action = agent.select_action(state, ou_noise)  # already flat, length=3

            # Reshape for env: each AP gets 1 null angle -> (num_bs, 1)
            action_env = action.reshape(env.num_bs, min(config.num_antennas - 1, env.num_bs - 1))

            next_obs, reward, done, _ = env.step(action_env)


            next_state = next_obs.flatten()

            # Store flat action (length=3) in replay buffer
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))

            episode_reward += reward
            state = next_state
            t += 1

            if done:
                break
        
        if i % 100 == 0:
            print("Action sample:", action)

        allRewards.append(episode_reward)

        # Fix 4: Only update after buffer is sufficiently filled
        if len(agent.replay_buffer.storage) >= MIN_BUFFER_SIZE:
            agent.update()

        if i % 100 == 0:
            avg = np.mean(allRewards[-100:])
            print(f"  Episode {i} | Steps: {t} | Reward: {episode_reward:.4f} | Avg(100): {avg:.4f}")
            agent.save()

    plt.plot(allRewards)
    plt.xlabel("Episode")
    plt.ylabel("Sum log2(rate)")
    plt.title("DDPG Learning Curve")
    plt.savefig("reward_curve.png")
    plt.show()



    code_end_time = time.time()
    elapsed_time = code_end_time - code_start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print the formatted elapsed time
    print(f"Elapsed Time for Code Execution: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")




    env.close()


