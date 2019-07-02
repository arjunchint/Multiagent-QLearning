print(1)
import numpy
import matplotlib.pyplot as plt
import time
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
		 [B.Ball*[64] + A_state*[8] + B_state, Actions]

Plotting: q_initial_s(A_South,B_Stay)
'''

gamma = 0.9
alpha = 0.3
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

	# Q table is size: states x actions => 112 possible states [2*8*7], 5 actions
	num_states = 128
	Q_A = numpy.zeros((num_states, 5))
	Q_B = numpy.zeros((num_states, 5))

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

		# Store prev-q to find difference
		prev_q = Q_A[record_state,1]
		prev_state = SoccerGame.B.Ball*(64) + (4*SoccerGame.A.y + SoccerGame.A.x)*8 + (4*SoccerGame.B.y + SoccerGame.B.x)*1

		# Use eps to determine actions?
		if numpy.random.uniform() < eps:
			A_action = numpy.random.randint(0,5)
			B_action = numpy.random.randint(0,5)
		else:
			A_action = numpy.argmax(Q_A[prev_state,:])
			B_action = numpy.argmax(Q_B[prev_state,:])

		# Rewards in terms of A, just negative for B
		rewards = SoccerGame.time_step(A_action,B_action)

		# Q update
		next_state = SoccerGame.B.Ball*(64) + (4*SoccerGame.A.y + SoccerGame.A.x)*8 + (4*SoccerGame.B.y + SoccerGame.B.x)*1

		max_Q_A = numpy.max(Q_A[next_state,:])
		max_Q_B = numpy.max(Q_B[next_state,:])

		if rewards:
			Q_A[prev_state,A_action] = (1-alpha)*Q_A[prev_state,A_action] + alpha * ((1-gamma)*rewards)
			Q_B[prev_state,B_action] = (1-alpha)*Q_B[prev_state,B_action] + alpha * ((1-gamma)*-rewards)			
		else:
			Q_A[prev_state,A_action] = (1-alpha)*Q_A[prev_state,A_action] + alpha * ((1-gamma)*rewards + gamma * max_Q_A)
			Q_B[prev_state,B_action] = (1-alpha)*Q_B[prev_state,B_action] + alpha * ((1-gamma)*-rewards + gamma *max_Q_B)

		# eps decay
		if eps > min_eps and iteration >1000:
			eps *= eps_decay	

		if alpha > 0.001:
			alpha *= eps_decay

		# Record Q value difference if there is change for plotting
		if prev_q != Q_A[record_state,1]:
			q_diff = abs(Q_A[record_state,1] - prev_q)
			sim_iter.append(iteration)
			plot_q.append(q_diff)
		# Sanity check
		if iteration%100000 == 0:
			elapsed_time=time.time()-start_time
			print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),"Iter: ",iteration,"/", iterations, "Q diff: ",q_diff,"Q_A: ",Q_A[record_state,0],"Eps: ", eps,'Aplha: ',alpha)
	print(Q_A)
	# Plot results
	plt.figure(1)
	plt.plot(sim_iter, plot_q)
	plt.xlabel('Simulation Iteration')
	plt.ylabel('Q-value Difference')
	plt.ylim(0,.5)
	plt.xlim(0,iterations)
	plt.show()
