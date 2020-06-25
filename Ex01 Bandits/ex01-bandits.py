import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward) #add new reward in the reward list
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)# 10 rewards position for 10 arms
    
    n_plays = np.zeros(bandit.n_arms)# 10 actions position for 10 arms. Each play will add 1 action time at correspondent position
    Q = np.zeros(bandit.n_arms) # 10 Q estimated values for 10 arms.
    possible_arms = range(bandit.n_arms)# Possible Choice number

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for arm in range (bandit.n_arms):
       rewards[arm] = bandit.play_arm(arm)
       n_plays[arm] = 1   
       Q[arm] = float(rewards[arm]/n_plays[arm])
    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        #a = random.choice(possible_arms)
        #reward_for_a = bandit.play_arm(a)
        # TODO: instead do greedy action selection
        # find Q-max
        a = np.argmax(Q) #find action number that has the maximum Q value
        reward_for_a = bandit.play_arm(a) #take this action and get the reward
        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        n_plays[a] += 1 #add 1 at correspondent postition
        rewards[a] += reward_for_a #add new rewards for correspondent action
        Q[a] = float(rewards[a] / n_plays[a]) #rewards/n_plays update correspondent Q value of action a



def epsilon_greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)# 10 rewards position for 10 arms
    n_plays = np.zeros(bandit.n_arms)# 10 actions position for 10 arms. Each play will add 1 action time at correspondent position
    Q = np.zeros(bandit.n_arms) # 10 Q estimated values for 10 arms.
    possible_arms = range(bandit.n_arms)# Possible Choice number
    epsilon = 0.1
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    while bandit.total_played < timesteps:
        epsilon_prob = np.random.uniform(0.,1.)
        if epsilon_prob > epsilon:
          #Q_max = max(Q) # find Q-max
          a = np.argmax(Q) #find action number that has the maximum Q value
          reward_for_a = bandit.play_arm(a) #take this action and get the reward
          # TODO: update the variables (rewards, n_plays, Q) for the selected arm
          n_plays[a] += 1 #add 1 at correspondent postition
          rewards[a] += reward_for_a #add new rewards for correspondent action
          Q[a] = float(rewards[a] / n_plays[a]) #rewards/n_plays update correspondent Q value of action a
        else:
          a = np.random.randint(low = 0, high = 10)
          reward_for_a = bandit.play_arm(a)
          # TODO: update the variables (rewards, n_plays, Q) for the selected arm
          n_plays[a] += 1 #add 1 at correspondent postition
          rewards[a] += reward_for_a #add new rewards for correspondent action
          Q[a] = float(rewards[a] / n_plays[a]) #rewards/n_plays update correspondent Q value of action a

def main():
    #n_episodes = 500  # TODO: set to 10000 to decrease noise in plot
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        #print(b._arm_means)
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()
if __name__ == "__main__":
    main()