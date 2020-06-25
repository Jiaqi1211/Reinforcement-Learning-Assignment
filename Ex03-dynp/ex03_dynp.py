# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:10:58 2020

@author: qinjiaqi
"""
import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    iterations = 100000
    for i in range(iterations):
        updated_V = np.copy(V_states)
        for state in range(n_states):
            Q_value = []
            for action in range(n_actions):
                next_states_rewards = []
                for p,n_state,r,is_terminal in env.P[state][action]:
                    next_states_rewards.append((p*(0+gamma*updated_V[n_state])))
                    Q_value.append(np.sum(next_states_rewards))
                    V_states[state] = max(Q_value)
        
        # if converge
        if(np.sum(np.fabs(updated_V - V_states)) <= theta):
            print("value_iteration converged at iteration # %d" % (i+1))
            break
    print("optimal value function is:",V_states)
    
    policy = np.zeros(n_states)
    for state in range(n_states):
        Q_table = np.zeros(n_states)
        for action in range(n_actions):
            for p,n_state,r,is_terminal in env.P[state][action]:
                Q_table[action] += (p*(r+gamma*V_states[n_state]))
        policy[state] = np.argmax(Q_table)
    return policy

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
