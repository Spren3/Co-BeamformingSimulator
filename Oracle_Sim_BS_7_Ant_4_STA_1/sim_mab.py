import csv
import numpy as np
import random
import helper
from simulator import Sim
from config import Config
from contextual_mab import ContinuousRandomBandit, OracleHeuristicBandit, NeuralBandit2

'''
Simulate different agents: random, oracle, ...

Author: Zubow (TU Berlin)
'''

def run_random_agent(config):
    run(0, config)

def run_oracle_agent():
    run(1)

def run_learn_agent(config):
    run(2, config)

def run(agent_type):
    print('run agent ...')
    seed = 42
    num_bss = 7
    num_ant = 4

    config = Config(num_bss, num_ant, seed)

    np.random.seed(config.seed)
    random.seed(config.seed)

    # create env
    env = Sim(config)

    if agent_type == 0:
        # create a random agent for each BS
        agents = [ContinuousRandomBandit(env.num_bs, config.num_antennas) for _ in range(env.num_bs)]
        s_prf = 'rnd'
    elif agent_type == 1:
        # oracle heuristic
        agents = [OracleHeuristicBandit(ii, env.num_bs, config.num_antennas) for ii in range(env.num_bs)]
        s_prf = 'oracle'
    elif agent_type == 2:
        # learning agent
        agents = [OracleHeuristicBandit(ii, env.num_bs, config.num_antennas) for ii in range(env.num_bs)]
        (n_actions, n_features) = env.get_spaces()
        agent = NeuralBandit2(n_actions, n_features)
        s_prf = 'learn'

    # run sim and collect rewards
    ep = 0
    rewards = np.zeros(60000)
    flag = True
    allRewards = []
    rewarddd = []
    done = True
    zz = 1
    print("BS: \t{}         Ant: \t{} ".format(num_bss, num_ant))
    
    for t in range(60000):
        if done:
            print(str(config.num_antennas) +'Antennas_DDPG_Noise/DDPG_Sim_BS_'+ str(env.num_bs) +'_Ant_' + str(config.num_antennas) + '_STA_' + str(config.max_num_stas))
            print(s_prf + ', t=%d, ep=%d, reset env' % (t, ep))
            if t != 0:
                
                for i in range(200):
                    ind = np.random.randint(0, 2000 * (zz), size=64)
                zz +=1
            obs = env.reset()
            state = obs.flatten()
            

            print("State1 ---------------->", state[0]*360)
            print("State2 ---------------->", state[1]*8)
            print("State3 ---------------->", state[2]*360)
            print("State4 ---------------->", state[3]*8)
            print("State5 ---------------->", state[4]*360)
            print("State6 ---------------->", state[5]*8)
            
            if t!=0:
                with open('H_All_Reward_BS_' + str(env.num_bs) + '_Antennas_' + str(config.num_antennas)+'_'+ str(config.max_num_stas)+'.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    if flag == True:
                        writer.writerow(['H_BS_' + str(env.num_bs) + '_Antennas_' + str(config.num_antennas)+ '_STA_' + str(config.max_num_stas)])
                        flag = False
                    for i in range(len(allRewards)):
                        writer.writerow([allRewards[i]])
                allRewards = []

        ep = t // config.max_steps_episode

        # predict from model
        if agent_type == 2:
            pred_rewards = agent.predict(obs)
        else:
            ind = np.random.normal()
            # print("here i am  ",t)
            pred_rewards = [rag.predict(obs) for rag in agents]
            

        # create action vector
        action = np.reshape(np.asarray(pred_rewards), (env.num_bs, min(config.num_antennas - 1, env.num_bs - 1)))
        

        # exec on env/bandit
        # print(action)
        obs, reward, done, _ = env.step(action)

        # print(obs)
        # env.render3()
        state = obs.flatten()
        if (t + 1) % 1000 == 0:
            stateee = obs.flatten()
            print("State1 ---------------->", stateee[0]*360)
            print("State2 ---------------->", stateee[1]*8)
            print("State3 ---------------->", stateee[2]*360)
            print("State4 ---------------->", stateee[3]*8)
            print("State5 ---------------->", stateee[4]*360)
            print("State6 ---------------->", stateee[5]*8)

        # print("State1 ---------------->", state[0]*360)
        # print("State2 ---------------->", state[1]*8)
        # print("State3 ---------------->", state[2]*360)
        # print("State4 ---------------->", state[3]*8)
        # print("State5 ---------------->", state[4]*360)
        # print("State6 ---------------->", state[5]*8)
        # rewards[t] = reward
        rewarddd.append(reward)

        # update models
        if agent_type == 2:
            agent.update(action, obs, reward)
        
        # print(config.max_steps_episode)
        if (t + 1) % config.max_steps_episode == 0:
            allRewards.append(round(sum(rewarddd)/len(rewarddd),2) )
            print(s_prf + ': mean of last %d-> %.2f' % (len(rewarddd), sum(rewarddd)/len(rewarddd)))
            rewarddd=[]

        
        

    # print('*** %s agent in (%s): mean reward %.2f' % (s_prf, config, np.mean(rewards)))

    # save to file
    # out_fname = 'data/' + s_prf + '_sim_mab_' + config.fname_str() + '.npy'
    # helper.write_results_to_file(out_fname, agents[0].agent_type, config, rewards)


if __name__ == '__main__':
    # quick test
    #cfg = Config(3, 7)
    cfg = Config(3, 3)
    cfg.max_steps_episode = 10
    cfg.n_steps = 50
    #run_random_agent(cfg)
    #run_oracle_agent(cfg)
    run_learn_agent(cfg)