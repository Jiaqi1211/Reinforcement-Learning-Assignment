import gym
import numpy as np
import matplotlib.pyplot as plt
import math

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
	""" TODO: implement the n-step sarsa algorithm """

	n_states = env.observation_space.n
	n_actions = env.action_space.n
	action = [0,1,2,3]
	Q_value = np.zeros((n_states,n_actions))
	V_value = np.zeros(n_states)
	hole = [19,29,35,41,42,46,49,52,54,59]
	goal = [63]
	rms_list = []

	for i in goal:
		#V_value[i] = 1
		Q_value[i][:]=1
	tau = 0

	for num in range(num_ep):
		#state = np.random.randint(n_states)
		#print (state)
		T = pow(10,20)
		#print(env.reset())
		env.reset()
		R_store = [0]
		S_store = [0]

		for t in range(n_states):
			if t < T:

				#act = epsilon_greedy(Q_value,t,action,epsilon)
				act = choose_action(action)
				new_state, reward, done, info = env.step(act)
				R_store.append(reward)
				#print(R_store)
				S_store.append(new_state)
				#print (new_state, reward, done, info)
				if done == True:
					T = t+1
			tau = t-n+1
			if tau >= 0:
				high_min = min(tau+n,T)
				low = tau + 1
				length = high_min - low + 1
				G = 0
				for j in range(length):
					G += pow(gamma,j)*R_store[j+tau+1]

				if tau+n < T:
					#print(tau+n)
					#G += pow(gamma,n)*V_value[S_store[tau+n]]
					G += pow(gamma, n) * Q_value[S_store[tau + n]][act]
				#V_value[S_store[tau]] += alpha*(G - V_value[S_store[tau]])
				Q_value[S_store[tau]][act] += alpha * (G - Q_value[S_store[tau]][act])
			if tau == T -1:
				#print("tau = ",tau)
				break
		rms_list.append(rms(Q_value))
	return Q_value, np.mean(rms_list)


def choose_action(actions):
	action = np.random.choice(actions)
	return action

def epsilon_greedy(Q,state,actions,epsilon):
	i = np.random.rand()
	if i < epsilon:
		action = np.random.choice(actions)
	else:
		action = np.argmax(Q[state][:])
	return action

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))




env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
diff_n = [1,2,4,8,16,32,64,128,256,512]
diff_a = [0,0.1,0.2,0.3,0.4,0.6,0.8,1]
#diff_n = [1,2]
#diff_a = [0.1,0.2]
p = {}
for i in diff_n:
	for j in diff_a:
		V, p[i,j] = nstep_sarsa(env,n=i,alpha=j)
print(p)
x_axis_n1 = []
y_axis_n1 = []
x_axis_n2 = []
y_axis_n2 = []
x_axis_n4 = []
y_axis_n4 = []
x_axis_n8 = []
y_axis_n8 = []
x_axis_n16 = []
y_axis_n16 = []
x_axis_n32 = []
y_axis_n32 = []
x_axis_n64 = []
y_axis_n64 = []
x_axis_n128 = []
y_axis_n128 = []
x_axis_n256 = []
y_axis_n256 = []
x_axis_n512 = []
y_axis_n512 = []


for key in p:
	if key[0] == 1:
		x_axis_n1.append(key[1])
		y_axis_n1.append(p[key])
	elif key[0] == 2:
		x_axis_n2.append(key[1])
		y_axis_n2.append(p[key])
	elif key[0] == 4:
		x_axis_n4.append(key[1])
		y_axis_n4.append(p[key])
	elif key[0] == 8:
		x_axis_n8.append(key[1])
		y_axis_n8.append(p[key])
	elif key[0] == 16:
		x_axis_n16.append(key[1])
		y_axis_n16.append(p[key])
	elif key[0] == 32:
		x_axis_n32.append(key[1])
		y_axis_n32.append(p[key])
	elif key[0] == 64:
		x_axis_n64.append(key[1])
		y_axis_n64.append(p[key])
	elif key[0] == 128:
		x_axis_n128.append(key[1])
		y_axis_n128.append(p[key])
	elif key[0] == 256:
		x_axis_n256.append(key[1])
		y_axis_n256.append(p[key])
	elif key[0] == 512:
		x_axis_n512.append(key[1])
		y_axis_n512.append(p[key])
print(x_axis_n1,y_axis_n1)
plt.plot(x_axis_n1,y_axis_n1,label = "n=1")
plt.plot(x_axis_n2,y_axis_n2,label = "n=2")
plt.plot(x_axis_n4,y_axis_n4,label = "n=4")
plt.plot(x_axis_n8,y_axis_n8,label = "n=8")
plt.plot(x_axis_n16,y_axis_n16,label = "n=16")
plt.plot(x_axis_n32,y_axis_n32,label = "n=32")
plt.plot(x_axis_n64,y_axis_n64,label = "n=64")
plt.plot(x_axis_n128,y_axis_n128,label = "n=128")
plt.plot(x_axis_n256,y_axis_n256,label = "n=256")
plt.plot(x_axis_n512,y_axis_n512,label = "n=512")
plt.plot.xlabel("alpha")
plt.plot.ylabel("RMS")
plt.legend()
plt.show()