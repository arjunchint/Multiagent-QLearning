print(1)
import numpy
import matplotlib.pyplot as plt
import time
import copy
from cvxopt import matrix, solvers,modeling

print(1)

''' 
Grid Visualization:

States: 8
	 x ->
	 0  1  2 3
y 0	|a 	B* A b|
| 1	|a  _  _ b|

	4y+x

Rewards: +100|-100
	
Actions:	N,S,E,W,Stay [5]
			0 1 2 3 4

Q_table: [2*8*7,5]
		 [B.Ball*[64] + A_state*[8] + B_state, Actions,Actions]

Plotting: q_initial_s(A_South,B_Stay)
'''

gamma = 0.9
alpha = 0.2
eps_decay = 0.999997
min_eps = 0.001
iterations = 1000000

print(1)
class SoccerWorld():
	def __init__(self):
		self.A = Agent(2,0,False)
		self.B = Agent(1,0,True)

	def act(self,action,current_pos):
		# North
		if action == 0 and current_pos[1]==1:
			current_pos[1] = 0
		# South				
		elif action == 1 and current_pos[1]==0:
			current_pos[1] = 1
		# East
		elif action == 2 and current_pos[0] != 3:
			current_pos[0] += 1
		# West
		elif action == 3 and current_pos[0] != 0:
			current_pos[0] -= 1
		# Stay
		elif action == 4:
			pass
		return current_pos

	def time_step(self,A_action,B_action):
		# [x,y]
		future_state_A = self.act(A_action,[self.A.x,self.A.y])
		future_state_B = self.act(B_action,[self.B.x,self.B.y])

		# Randomly choose if player A goes first
		first_player = numpy.random.randint(0,2)

		# A goes first  
		if first_player:
			# A and B run into each other's current states: no move, maybe change own
			if future_state_A == [self.B.x,self.B.y] and future_state_B == [self.A.x,self.A.y]:
				if self.A.Ball:
					pass
				else:
					self.A.Ball,self.B.Ball = True, False		
			# A runs into B: if A has ball switch ownership, else nothing. B moves
			elif future_state_A == [self.B.x,self.B.y]:
				if self.A.Ball:
					self.A.Ball,self.B.Ball = False, True
				else:
					pass
				self.B.x, self.B.y = future_state_B[0], future_state_B[1]
			# A and B move to same future location but A moves first so A gets ball AND moves to loc
			elif future_state_A == future_state_B:
				if self.A.Ball:
					pass
				else:
					self.A.Ball,self.B.Ball = True, False
				self.A.x, self.A.y = future_state_A[0], future_state_A[1]
			# A and B both move two different locations
			else:
				self.A.x, self.A.y = future_state_A[0], future_state_A[1]
				self.B.x, self.B.y = future_state_B[0], future_state_B[1]

		# B goes first
		else:
			# A and B run into each other's current states: no move, maybe change own
			if future_state_A == [self.B.x,self.B.y] and future_state_B == [self.A.x,self.A.y]:
				if self.A.Ball:
					self.A.Ball,self.B.Ball = False, True
				else:
					pass
			# B runs into A: if B has ball switch ownership, else nothing
			elif future_state_B == [self.A.x,self.A.y]:
				if self.B.Ball:
					self.A.Ball,self.B.Ball = True, False
				else:
					pass
				self.A.x, self.A.y = future_state_A[0], future_state_A[1]
			# A and B move to same future location but B moves first so B gets ball AND moves to loc
			elif future_state_A == future_state_B:
				if self.A.Ball:
					self.A.Ball,self.B.Ball = False, True
				else:
					pass
				self.B.x, self.B.y = future_state_B[0], future_state_B[1]
			# A and B both move two different locations
			else:
				self.A.x, self.A.y = future_state_A[0], future_state_A[1]
				self.B.x, self.B.y = future_state_B[0], future_state_B[1]
		

		# Goal Conditions
		if self.A.Ball:
			# GOAAAAAAAL: A Reward += 100, B Reward -= 100
			if self.A.x == 0:
				return 100
			# OwnGoal :( A Reward -= 100, B Reward += 100
			elif self.A.x == 3:
				return -100
			else:
				return 0
		elif self.B.Ball:
			# GOAAAAAAAL: A Reward -= 100, B Reward += 100
			if self.B.x == 3:
				return -100
			# OwnGoal :( A Reward += 100, B Reward -= 100
			elif self.B.x == 0:
				return 100				
			else:
				return 0


class Agent():
	def __init__(self,x,y,Ball):
		self.x = x
		self.y = y
		self.Ball = Ball


if __name__ == "__main__":
	eps = 1.0
	rewards_A=[]
	rewards_B=[]

	# Q table is size: states x actions x actions => 128 possible states [2*8*8], 5 actions, 5 actions
	n_states = 128
	n_actions = 5
	Q_A = numpy.ones((n_states, n_actions,n_actions),dtype=numpy.dtype(float))
	solvers.options['show_progress'] = False
	# Q_B = numpy.zeros((n_states, n_actions,n_actions))

	sim_iter=[]
	plot_q=[]

	rewards = 0
	SoccerGame = SoccerWorld()
	start_time=time.time()
	q_diff = 0

	record_state = SoccerGame.B.Ball*(64) + (4*SoccerGame.A.y + SoccerGame.A.x)*8 + (4*SoccerGame.B.y + SoccerGame.B.x)*1
	print('Recording State: ',record_state)
	for iteration in range(iterations):
		# Make sure only one player has ball
		assert SoccerGame.A.Ball + SoccerGame.B.Ball == 1

		# Make sure no player on same spot
		assert [SoccerGame.A.x,SoccerGame.A.y] != [SoccerGame.B.x,SoccerGame.B.y] 

		# Reset Game, if non zero rewards aka game finishes
		if rewards:
			SoccerGame = SoccerWorld()

		# Store prev-q to find difference with A South and B Stick
		prev_q = Q_A[record_state,1,4]
		prev_state = SoccerGame.B.Ball*(64) + (4*SoccerGame.A.y + SoccerGame.A.x)*8 + (4*SoccerGame.B.y + SoccerGame.B.x)*1

		# Use random actions?
		A_action = numpy.random.randint(0,5)
		B_action = numpy.random.randint(0,5)

		# Rewards in terms of A, just negative for B
		rewards = SoccerGame.time_step(A_action,B_action)

		# Q update
		next_state = SoccerGame.B.Ball*(64) + (4*SoccerGame.A.y + SoccerGame.A.x)*8 + (4*SoccerGame.B.y + SoccerGame.B.x)*1

		# Setup constraints to find minimax Q value

		# action_probabilities,constraints = [],[]
		# for action in range(n_actions):
		# 	action_probabilities.append(modeling.variable())
		# 	constraints.append(action_probabilities[action] >= 0)
		# constraints.append((sum(action_probabilities) == 1))

		action_probabilities,constraints = modeling.variable(5),[]
		constraints.append((action_probabilities >= 0))
		constraints.append((sum(action_probabilities) == 1))

		optimal = modeling.variable()

		# constraints.append((modeling.dot(action_matrix.T,action_probabilities)>= optimal))

		for action_B in range(n_actions):
			q_constr = 0
			for action_A in range(n_actions):
				q_constr += float(Q_A[next_state, action_A, action_B]) *action_probabilities[action_A]
				# q_constr += action_matrix[action_A, action_B] *action_probabilities[action_A]
			constraints.append((q_constr >= optimal))
		# print(constraints)
		# maximize the Q so negative of min
		lp = modeling.op(-optimal, constraints)
		lp.solve(solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
		# print(lp.status)
		max_Q = optimal.value[0]
		# print(max_Q)

		if rewards:
			Q_A[prev_state,A_action,B_action] = (1-alpha)*Q_A[prev_state,A_action,B_action] + alpha * ((1-gamma)*rewards)
			# Q_B[prev_state,B_action] = (1-alpha)*Q_B[prev_state,B_action] + alpha * ((1-gamma)*-rewards)			
		else:
			Q_A[prev_state,A_action,B_action] = (1-alpha)*Q_A[prev_state,A_action,B_action] + alpha * ((1-gamma)*rewards + gamma * max_Q)
			# Q_B[prev_state,B_action] = (1-alpha)*Q_B[prev_state,B_action] + alpha * ((1-gamma)*-rewards + gamma *max_Q_B)

		# eps decay
		if eps > min_eps and iteration > 1000:
			eps *= eps_decay	

		if alpha > 0.001:
			alpha *= eps_decay

		# Record Q value difference if there is change for plotting
		if prev_q != Q_A[record_state,1,4]:
			q_diff = abs(Q_A[record_state,1,4] - prev_q)
			sim_iter.append(iteration)
			plot_q.append(q_diff)

		# Sanity check
		if iteration%100000 == 0:
			elapsed_time=time.time()-start_time
			print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),"Iter: ",iteration,"/", iterations, "Q diff: ",q_diff,"Q_A: ",Q_A[record_state,1],"Eps: ", eps,'Aplha: ',alpha)
			
	# Plot results
	plt.figure(1)
	plt.plot(sim_iter, plot_q)
	plt.xlabel('Simulation Iteration')
	plt.ylabel('Q-value Difference')
	plt.ylim(0,.5)
	plt.xlim(0,iterations)
	plt.show()
