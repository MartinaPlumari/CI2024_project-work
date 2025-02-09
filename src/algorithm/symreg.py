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
import utils.draw as draw
import matplotlib.pyplot as plt
from itertools import accumulate


class Symreg:
	# data loading / initialization
	# counter 
	# cost (using the fitness function inside tree)

	problem : Problem
	history : list[float]

	# definire una classe quindi un tipo

	population : list[t.Tree] = list()
	use_validation : bool

	class MUTATION(Enum):
		SUBTREE = 0
		POINT = 1
		PERMUT = 2
		HOIST = 3
		EXPANSION = 4
		COLLAPSE = 5
	
	class POPULTAION_MODEL(Enum):
		STEADY_STATE = 0,
		GENERATIONAL = 1
	
	MUTATION_TYPE : MUTATION
	POP_MODEL : MUTATION
	POPULATION_SIZE : int
	OFFSPRING_SIZE : int
	MAX_GENERATIONS : int
	MUTATION_PROBABILITY : float
	TOURNAMENT_SIZE : int
	USE_RAND_MUT_TYPE : bool
	
	
	# fare in modo che in init vengano passati gli argmenti da linea di comando per decidere i vari iper parametri 
	# e le strategie come ad esempio quale tipo di mutazione usare.

	def __init__(self, 
			  problem : Problem, 
			  population_size : int = 100, 
			  offspring_size : int = 1_000, 
			  max_generations : int = 1_000,
			  mutation_type : MUTATION = MUTATION.POINT,
			  population_model : POPULTAION_MODEL = POPULTAION_MODEL.STEADY_STATE,
			  mutation_probability : float = 0.05,
			  tournament_size : int = 3,
			  use_random_mutation_type : bool = False) -> None:
		
		# initalize variables
		self.POPULATION_SIZE = population_size
		self.OFFSPRING_SIZE = offspring_size
		self.MAX_GENERATIONS = max_generations
		self.MUTATION_TYPE = mutation_type
		self.POP_MODEL = population_model
		self.MUTATION_PROBABILITY = mutation_probability
		self.TOURNAMENT_SIZE = tournament_size
		self.USE_RAND_MUT_TYPE = use_random_mutation_type

		self.history = list() 

		# extract problem data
		self.problem = problem
		self.use_validation = problem.use_validation_set
		self.mutation_type = mutation_type
		x = problem.x_train
		y = problem.y_train

		# init population
		for _ in range(self.POPULATION_SIZE):
			self.population.append(t.Tree(x, y, INIT_METHOD=t.INIT_METHOD.FULL))
	
	#tournament selection without replacement (try with replacement)
	def _parent_selection(self, population : list[t.Tree]):
		tournament_contestants = np.random.choice(population, self.TOURNAMENT_SIZE, replace=False) 
		best_candidate : t.Tree = max(tournament_contestants, key=lambda x: x._fitness) 
		return best_candidate##.deep_copy()
 	
	def _mutation(self, individual : t.Tree) -> t.Tree:
		"""
		Select and apply a mutation to an individual

		:param individual:
		:type individual: t.Tree

		:param mut_type: The mutation used on the individual 
		:type mut_type: MUTATION(enum)

		:param use_random_mutation_type: If true ignore the mut_type and select<br/>randomly a mutation type with eaven<br/>probability.
		:type use_random_mutation_type: bool
		"""

		mut_type = self.MUTATION_TYPE

		if self.USE_RAND_MUT_TYPE:
			# select random mutation type
			mut_type = random.choice(list(self.MUTATION))

		# select mutation type
		match mut_type:
			case self.MUTATION.SUBTREE:
				if individual._h > 3: 
					individual = t.subtree_mutation(individual)
				else:
					individual = t.expansion_mutation(individual)
			case self.MUTATION.POINT:
				individual = t.point_mutation(individual)
			case self.MUTATION.PERMUT:
				individual = t.permutation_mutation(individual)
			case self.MUTATION.HOIST:
				if individual._h <= 4: 
					individual = t.hoist_mutation(individual)
				else:
					individual = t.collapse_mutation(individual)
			case self.MUTATION.COLLAPSE:
				if individual._h > 3: 
					individual = t.collapse_mutation(individual)
				else:
					individual = t.expansion_mutation(individual)
			case self.MUTATION.EXPANSION:
				if individual._h <= 4: 
					individual = t.expansion_mutation(individual)
				else:
					individual = t.collapse_mutation(individual)

		return individual


	def _step(self, population : list[t.Tree]) -> None:
		"""
		Use hyper modern approach
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
				
		for child in offspring:
			child._fitness = child.fitness
		
		match self.POP_MODEL:
			case self.POPULTAION_MODEL.STEADY_STATE, default:
				population.extend(offspring)
				population.sort(key=lambda i : i._fitness, reverse=False)
			case self.POPULTAION_MODEL.GENERATIONAL:
				population = sorted(offspring, key=lambda i : i._fitness, reverse=False)
			
		population = population[:self.POPULATION_SIZE]

		return population
	
	def train(self) -> None:
		current_population : list[t.Tree] = self.population
		best_solution : t.Tree = current_population[0].deep_copy()
		
		last_fitness : float = 0
		steady_counter : int = 0

		self.history.append(max(current_population, key= lambda x: x._fitness)._fitness)

		for i in range(self.MAX_GENERATIONS):
			current_population = self._step(current_population)
			
			current_solution = max(current_population, key= lambda x: x._fitness)
			
			self.history.append(current_solution._fitness)
			
			if best_solution._fitness < current_solution._fitness:
				best_solution = current_solution.deep_copy()
			
			if i % 50 == 0:
				if best_solution._fitness == last_fitness:
					steady_counter += 1
					if steady_counter > 3:
						self.MUTATION_PROBABILITY += 0.1
					if steady_counter > 5:
						break
				else:
					last_fitness = best_solution._fitness
					self.MUTATION_PROBABILITY = 0.05
					# if self.MUTATION_PROBABILITY < 0.05:
					# 	self.MUTATION_PROBABILITY = 0.05
					steady_counter = 0
				print(f"STEP [{i}/{self.MAX_GENERATIONS}] || fitness : {best_solution._fitness} || {best_solution._root.long_name}")

			if steady_counter > 50:
				self.MUTATION_PROBABILITY += 0.05
		
		self.history.append(best_solution._fitness)
		self.problem.solution = best_solution
	
	def plot_history(self) -> None:
		plt.close('all')
		plt.figure(figsize=(8, 6))
		plt.plot(
			range(len(self.history)),
			list(accumulate(self.history, max)),
			color="red"
		)
		_ = plt.scatter(
			range(len(self.history)),
			self.history, 
			marker = '.'
		)
		plt.show()