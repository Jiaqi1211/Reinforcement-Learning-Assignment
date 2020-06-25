
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product




env = gym.make('MountainCar-v0')

#create discrete state space
state_space = np.linspace( env.observation_space.low, env.observation_space.high, 20).T
states = np.array(list(product(*state_space)))
#number of states and actions
n_states = 400
n_actions = 3  
# get state from env observation
def get_state(obs):
    return np.argmin(np.linalg.norm(obs-states,axis=1))

# define Q(lambda), number of episodes=5000, episode length=200    
def qlearning_lambda(env, alpha=0.1, gamma=0.9, epsilon=0.9, num_ep=5000, lamda=0.9, ep_len=200):
    
    # define epsilon-greedy  policy
    def epsilon_greedy(Q, state):        
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()          
        else:
            action = np.argmax(Q[state,:])
        return action    
    # define greedy policy
    def greedy(Q, state):
        action = np.argmax(Q[state,:])
        return action    
      
    #value for states and  actions
    Q = np.random.random((n_states,n_actions))     
    # Eligibility traces
    Elig = np.zeros((n_states,n_actions))
    # rewards per episode
    reward_list = []
    ave_reward_list = []   
    # number of successes and steps per episode
    num_successes = []
    avg_successes = []
    num_steps = []
    avg_steps = []
    
    for i in range(num_ep):
        done = False
        obs = env.reset()
        s = get_state(obs)
        total_reward = 0
     
        for j in range(ep_len):
            if (i+1) % 100 == 0:
                env.render()
            
            # choose action with epsilon-greedy policy 
            a = epsilon_greedy(Q,s) 
            obs_, reward, done, info = env.step(a)
            s_ = get_state(obs_)
            # choose next action from next state with greedy policy
            a_ = greedy(Q, s_) 
            # update reward
            total_reward += reward
            # keep an eligibility trace
            Elig *= lamda * gamma 
            Elig[s,a] += 1
            #update Temporal Difference 
            TD = reward + gamma*Q[s_,a_] - Q[s,a] 
            #update value 
            Q += alpha*TD*Elig 
            #update state
            s = s_
                       
            if done:
                
                break
        #decay epsilon    
        epsilon *= 0.99 
        #        
        reward_list.append(total_reward)
        
        num_steps.append(j)
        
        if obs_[0] >= 0.5:
            num_successes.append(1)
        else:
            num_successes.append(0)   
        
        if (i+1) % 100 == 0:
             avg_success = np.mean(num_successes)
             avg_successes.append(avg_success)
             num_successes = []
             avg_step = np.mean(num_steps)
             avg_steps.append(avg_step)
             num_steps = []
             ave_reward = np.mean(reward_list)
             ave_reward_list.append(ave_reward)
             reward_list = []

    return avg_successes,avg_steps,ave_reward_list


    
    
def main():
    env.reset()
    avg_suc,avg_step,rewards= qlearning_lambda(env)  
    plt.plot(100*(np.arange(len(avg_suc)) + 1), avg_suc,label = "average success")
    plt.plot(100*(np.arange(len(avg_step)) + 1), avg_step, label = "average steps")
    plt.xlabel('Episodes')
    plt.ylabel("success rate")
    plt.legend()    
    plt.figure()
    
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
