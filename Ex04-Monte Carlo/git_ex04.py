#%matplotlib inline

import gym
import matplotlib
import numpy as np
import sys
import plotting

from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from envs.blackjack import BlackjackEnv

matplotlib.style.use('ggplot')

env = gym.make('Blackjack-v0')


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):


    # 记录每个状态的Return和出现的次数。
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # 最终的价值函数
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 生成一个episode.
        # 一个episode是三元组(state, action, reward) 的数组
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # 找到这个episode里出现的所有状态。
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # 找到这个状态第一次出现的下标
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # 计算这个状态的Return
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # 累加
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V
def sample_policy(observation):
    """
    一个简单的策略：如果不到20就继续要牌，否则就停止要牌。
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def plot_3d(Value_data,useable_ace):
    V_x = []
    V_y = []
    V_z = []
    #print("")
    #print(V_10k)
    if useable_ace == 1:
        for ii in Value_data.items():
            if ii[0][2] == True:
                V_x.append(ii[0][0])
                V_y.append(ii[0][1])
                V_z.append(ii[1])
    else:
        for ii in Value_data.items():
            if ii[0][2] == False:
                V_x.append(ii[0][0])
                V_y.append(ii[0][1])
                V_z.append(ii[1])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(V_x, V_y, V_z)
    plt.show()

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plot_3d(V_10k,1)
plot_3d(V_10k,0)


V_500k = mc_prediction(sample_policy, env, num_episodes=500000)

plot_3d(V_500k,1)
plot_3d(V_500k,0)

