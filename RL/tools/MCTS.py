import numpy as np
import pandas as pd
from copy import deepcopy
import random


class MCTS(object):
	def __init__(self, simulator=None, rollout_policy_model=None, max_nb_sims=1000, mcts_timeout=None,
	             current_state=None,
	             max_depth=5):

		self.default_simulator = simulator
		self.simulator = None
		self.rollout_policy_model = rollout_policy_model
		self.max_nb_sims = max_nb_sims
		self.mcts_timeout = mcts_timeout
		self.current_state = current_state
		self.max_depth = max_depth
		self.nb_sims = 0

		# self.search_tree = pd.DataFrame(columns=['parent_index', 'state', 'children_index','action', 'Q_value',
		#                                          'rewards',
		#                                          'nb_par_sel',
		#                                          'nb_sel']) # stats
		# self.selection_traj = None

	def rescale_reward(self, reward):
		if reward != 0:
			return (reward+100)/300
		else:
			return 0

	def uct(self):
		""" Conduct a UCT search for itermax iterations starting from rootstate.
			Return the best move from the rootstate.
			Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

		rootnode = Node()
		depth = 0

		for i in range(self.max_nb_sims):
			node = rootnode
			self.simulator = deepcopy(self.default_simulator)


			# Select
			while node.untriedActions == [] and node.childNodes != [] and depth < self.max_depth: # node is fully
				# expanded and
				# non-terminal
				node = node.uct_select_child()
				self.simulator.step(node.action)
				depth += 1

			# Expand
			if node.untriedActions != []: # if we can expand (i.e. state/node is non-terminal)
				action = random.choice(node.untriedActions)
				next_state, reward, is_terminal, _ = self.simulator.step(action)
				node = node.add_child(action, deepcopy(next_state)) # add child and descend tree

			# Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
			done = False
			sum_reward = 0

			t = 0
			discount_factor = 0.98

			while not done:
				q_values = self.rollout_policy_model.predict_on_batch(np.reshape(next_state,(1,1,85)))
				action = np.argmax(q_values)
				next_state, reward, is_terminal, _ = self.simulator.step(action)
				sum_reward += self.rescale_reward(np.power(discount_factor, t)*reward)
				t += 1
				if is_terminal:
					done = True

			# Backpropagate
			while node is not None: # backpropagate from the expanded node and work back to the root node
				node.update(sum_reward) # state is terminal. Update node with result from POV of node.playerJustMoved
				node = node.parentNode

		# return sorted(rootnode.childNodes, key=lambda c:c.visits)[-1].action # return the move that was most visited
		return sorted(rootnode.childNodes, key=lambda c:np.mean(c.rewards))[-1].action # return the move that
	# was
	# most
	# visited


class Node(MCTS):
	def __init__(self, action=None, parent=None, state=None, **kwargs):
		super(Node, self).__init__(**kwargs)
		self.action = action # the move that got us to this node - "None" for the root node
		self.parentNode = parent # "None" for the root node
		self.state = state
		self.childNodes = []
		# self.rewards = 0
		self.rewards = []
		self.visits = 0
		self.untriedActions = [0, 1, 2]

	def uct_select_child(self):
		""" Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
			lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
			exploration versus exploitation.
		"""
		# s =\
		# 	sorted(self.childNodes,
		# 	       key=lambda c:c.rewards / c.visits + np.sqrt(2 * np.log(self.visits) / c.visits))[-1]

		s =\
			sorted(self.childNodes,
			       key=lambda c: np.mean(c.rewards) + np.sqrt(2 * np.log(self.visits) / c.visits))[-1]
		return s

	def add_child(self, m, s):
		""" Remove m from untriedMoves and add a new child node for this move.
			Return the added child node
		"""
		n = Node(action=m, parent=self, state=s)
		self.untriedActions.remove(m)
		self.childNodes.append(n)
		return n

	def update(self, reward):
		""" Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
		"""
		self.visits += 1
		# self.rewards += reward
		self.rewards.append(reward)








	# def build_tree(self):
	# 	# build "tree" for implementing UCT and backup
	# 	# e.g. each node (parent_index, state, children_index, action, Q_value, nb_par_sel, nb_sel)
	# 	# child
	# 	# selections))
	#
	# 	# all nodes are status, the "state" field of each entry
	#
	# 	self.search_tree.loc[0] = [[], self.current_state, [], 0, 0, None, 0]
	#
	# def apply_tree_policy(self, node):
	# 	# use UCT to return an entry of search tree, cp=1
	# 	# select one child of given node
	#
	# 	children = self.search_tree.find('index' == node['children_index'])
	#
	# 	UCT = children['Q_value'] + np.sqrt(2*np.log(children['nb_par_sel'])/children['nb_sel'])
	# 	child = np.argmax(UCT)
	# 	return child
	#
	# def selection(self):
	# 	# tree policy is UCT
	# 	# select child based on UCT (calculated by Q and states)
	# 	# select ends when a leaf(no children) is selected
	# 	depth = 0
	# 	node = self.search_tree[self.current_state] # an entry, the root
	#
	# 	node = self.search_tree.loc[self.search_tree['state'] == self.current_state]
	#
	# 	self.selection_traj.append([node])
	# 	while depth < self.max_depth:
	#
	# 		if len(self.search_tree) == 1: # could't select at depth root, only one node at this depth, go to expansion
	# 			# directly
	# 			break
	#
	# 		node = self.apply_tree_policy(node) # return the selected child
	#
	# 		self.simulator.step(node['action']) # step the env
	#
	# 		depth += 1
	# 		self.selection_traj.append([node])
	#
	# 		if not node['children_index']:
	# 			break
	#
	# 	return node
	#
	# def expansion(self, leaf):
	# 	# add possible entries to the search tree
	# 	# since we only have three actions per state, for now we expand tree by all three actions
	# 	# create 3 children
	# 	# append to the three
	# 	children = []
	# 	for i in range(self.simulator.num_action):
	# 		parent_index = self.search_tree[leaf].index
	# 		parent_sel = self.search_tree[leaf]['nb_sel']
	# 		next_state, reward, is_terminal, _ = self.simulator.step(i)
	# 		child = [parent_index, next_state, [], i, 0, [], parent_sel, 0]
	# 		self.search_tree.append(child)
	# 		self.search_tree[parent_index]['children_index'].append(self.search_tree['index'][-1])
	# 		children.append(child)
	#
	# 	return children
	#
	# def run_simulation(self, child):
	# 	# run simulations on expanded children
	# 	done = False
	# 	sum_reward = 0
	# 	state = child['state']
	# 	while not done:
	# 		q_values = self.rollout_policy(state)
	# 		action = np.argmax(q_values)
	# 		next_state, reward, is_terminal, _ = self.simulator.step(action)
	# 		sum_reward += reward
	# 		if is_terminal:
	# 			done = True
	#
	# 	sim_traj = deepcopy(self.selection_traj)
	# 	sim_traj.append(child)
	#
	# 	# TODO: rescale reward between 0 and 1
	# 	self.backup(sim_traj, sum_reward)
	#
	# def backup(self, sim_traj, sum_reward):
	# 	# update Q values of expanded children, their parents and parents of parents until root
	# 	# update the stats of all leaves and nodes
	# 	for node in sim_traj:
	# 		if node['nb_par_sel']:
	# 			node['nb_par_sel'] += 1
	# 		node['nb_sel'] += 1
	# 		node['rewards'].append(sum_reward)
	# 		node['Q_value'] = np.mean(node['rewards'])
	#
	# def return_action(self):
	# 	# execute MCTS, get the optimal action based on the highest Q
	# 	self.build_tree()
	# 	while self.nb_sims < self.max_nb_sims:
	# 		self.selection_traj = []
	# 		self.simulator = deepcopy(self.default_simulator)
	# 		leaf = self.selection()
	# 		rollout_children = self.expansion(leaf)
	#
	# 		for child in rollout_children:
	# 			self.run_simulation(child)
	# 			self.nb_sims += 1
	#
	# 	return self.final_action()
	#
	# def final_action(self):
	# 	# final selection is based on highest Q value of all children of cur_state
	# 	action = np.argmax(self.search_tree['Q_value'])
	# 	return action

