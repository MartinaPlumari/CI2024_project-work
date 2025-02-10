# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import tree.tree as t
from utils.problemloader import Problem
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate


class Symreg:
	"""
	Class that handle training for a symbolic regression problem.
	"""

	problem : Problem
	history : list[float]
	population : list[t.Tree] = list()
	use_validation : bool

	class MUTATION(Enum):
		SUBTREE = 0
		POINT = 1
		PERMUT = 2
		HOIST = 3
		EXPANSION = 4
		COLLAPSE = 5
	
	class POPULATION_MODEL(Enum):
		STEADY_STATE = 0,
		GENERATIONAL = 1
  
	class INIT_METHOD(Enum):
		GROW = 0
		FULL = 1
		HALF_HALF = 2
	
	MUTATION_TYPE : MUTATION
	POP_MODEL : POPULATION_MODEL
	POP_INIT_METHOD : INIT_METHOD
	POPULATION_SIZE : int
	OFFSPRING_SIZE : int
	MAX_GENERATIONS : int
	MUTATION_PROBABILITY : float
	TOURNAMENT_SIZE : int
	USE_RAND_MUT_TYPE : bool

	def __init__(self, 
			  problem : Problem, 
			  population_size : int = 100, 
			  offspring_size : int = 1_000, 
			  max_generations : int = 1_000,
			  mutation_type : MUTATION = MUTATION.POINT,
			  population_model : POPULATION_MODEL = POPULATION_MODEL.STEADY_STATE,
			  population_init_method : INIT_METHOD = INIT_METHOD.HALF_HALF,
			  mutation_probability : float = 0.05,
			  tournament_size : int = 3,
			  use_random_mutation_type : bool = False) -> None:
		
		# initalize variables
		self.POPULATION_SIZE = population_size
		self.OFFSPRING_SIZE = offspring_size
		self.MAX_GENERATIONS = max_generations
		self.MUTATION_TYPE = mutation_type
		self.POP_MODEL = population_model
		self.POP_INIT_METHOD = population_init_method
		self.MUTATION_PROBABILITY = mutation_probability
		self.TOURNAMENT_SIZE = tournament_size
		self.USE_RAND_MUT_TYPE = use_random_mutation_type

		self.history = list() 
		self.population = list()

		# extract problem data
		self.problem = problem
		self.use_validation = problem.use_validation_set
		self.mutation_type = mutation_type

		self.init_population()
	
	def init_population(self) -> None:
		"""
		Initilize population generating random trees.
		"""
		x = self.problem.x_train
		y = self.problem.y_train

		# init population
		if self.POP_INIT_METHOD == self.INIT_METHOD.HALF_HALF:
			for _ in range(self.POPULATION_SIZE//2):
				self.population.append(t.Tree(x, y, INIT_METHOD=self.INIT_METHOD.FULL))
				self.population.append(t.Tree(x, y, INIT_METHOD=self.INIT_METHOD.GROW))
		else:
			for _ in range(self.POPULATION_SIZE):
				self.population.append(t.Tree(x, y, INIT_METHOD=self.POP_INIT_METHOD))
	
	def _parent_selection(self, population : list[t.Tree]):
		"""
		Simple tournament-based parent selection.
		"""
		tournament_contestants = random.sample(population, self.TOURNAMENT_SIZE) 
		best_candidate : t.Tree = max(tournament_contestants, key=lambda x: x._fitness) 
		return best_candidate
 	
	def _mutation(self, individual : t.Tree) -> t.Tree:
		"""
		Select and apply a choosen mutation to an individual.
		"""

		mut_type = self.MUTATION_TYPE

		if individual._h < 2:
			# if the tree is too small try expanding the tree
			individual = t.expansion_mutation(individual)
			return individual
		elif individual._h >= 4:
			# if the tree is too deep try reducing its size
			individual = t.hoist_mutation(individual)
			return individual

		if self.USE_RAND_MUT_TYPE:
			# randomize mutation type
			mut_type = random.choice(list(self.MUTATION))

		# select mutation
		match mut_type:
			case self.MUTATION.SUBTREE:
				individual = t.subtree_mutation(individual)
			case self.MUTATION.POINT:
				individual = t.point_mutation(individual)
			case self.MUTATION.PERMUT:
				individual = t.permutation_mutation(individual)
			case self.MUTATION.HOIST:
				individual = t.hoist_mutation(individual)
			case self.MUTATION.COLLAPSE:
				individual = t.collapse_mutation(individual)
			case self.MUTATION.EXPANSION:
				individual = t.expansion_mutation(individual)

		return individual


	def _step(self, population : list[t.Tree]) -> None:
		"""
		Use hyper modern approach for generating the offspring by randomly choosig whether to use mutation or recombination.<br/>
		Then extend the current population using a given population model.
		"""
		offspring : list[t.Tree] = list()

		for _ in range(self.OFFSPRING_SIZE):
			if np.random.random() <= self.MUTATION_PROBABILITY:
				# MUTATION
				p : t.Tree = self._parent_selection(population)
				o : t.Tree = self._mutation(p)

			else:
				# RECOMBINATION
				p1 : t.Tree = self._parent_selection(population)
				p2 : t.Tree = self._parent_selection(population)
				
				o : t.Tree = t.recombination(p1, p2)
			
			offspring.append(o)
		
		# update offspring fitenss
		for child in offspring:
			child._fitness = child.fitness
		
		# select population model and extend population
		match self.POP_MODEL:
			case self.POPULATION_MODEL.STEADY_STATE, default:
				population.extend(offspring)
				population.sort(key=lambda i : i._fitness, reverse=False)
			case self.POPULATION_MODEL.GENERATIONAL:
				population = sorted(offspring, key=lambda i : i._fitness, reverse=False)
		
		# resize population to max population
		population = population[:self.POPULATION_SIZE]

		return population
	
	def train(self) -> None:
		"""
		Uses a genetic programming algorithm to estimate a mathematical function that can solve a symbolic regression problem.
		"""

		# initialize population and solution
		current_population : list[t.Tree] = self.population
		best_solution : t.Tree = current_population[0]		
		last_fitness : float = 0
		steady_counter : int = 0

		self.history.append(max(current_population, key= lambda x: x._fitness)._fitness)

		for i in range(self.MAX_GENERATIONS):
			# tweak current solution
			current_population = self._step(current_population)
			
			# get the best tree from current population
			current_solution = max(current_population, key= lambda x: x._fitness)
			
			# append fitness value
			self.history.append(current_solution._fitness)
			
			# update best solution
			if best_solution._fitness < current_solution._fitness:
				best_solution = current_solution.deep_copy()
			
			# every 50 iterations
			if i % 50 == 0:
				# checks if the best solution remained unchanged
				if best_solution._fitness == last_fitness:
					steady_counter += 1
					if steady_counter > 3:
						# increase mutation probability
						self.MUTATION_PROBABILITY += 0.05
						if self.MUTATION_PROBABILITY > 1:
							self.MUTATION_PROBABILITY = 1
					if steady_counter > 6:
						# if it takes too long for improving then stop early
						break
				else:
					# reset
					last_fitness = best_solution._fitness
					self.MUTATION_PROBABILITY = 0.05
					steady_counter = 0
				
				# print intermediate state
				print(f"STEP [{i}/{self.MAX_GENERATIONS}] || fitness : {best_solution._fitness} || {best_solution._root.long_name}")
		
		self.history.append(best_solution._fitness)
		
		# save the solution
		self.problem.solution = best_solution
	
	def plot_history(self) -> None:
		"""
		Plot the current history. Usually history starts empty and get filled up during training.
		"""
		plt.close('all')
		plt.figure(figsize=(8, 6))
		plt.plot(
			range(len(self.history)),
			list(accumulate(self.history, max)),
			color="red"
		)
		plt.show()