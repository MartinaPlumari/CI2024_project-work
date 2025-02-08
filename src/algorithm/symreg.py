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


class Symreg:
	# data loading / initialization
	# counter 
	# cost (using the fitness function inside tree)

	problem : Problem

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
	
	MUTATION_TYPE : MUTATION
	POPULATION_SIZE : int
	OFFSPRING_SIZE : int
	MAX_GENERATIONS : int
	MUTATION_PROBABILITY : float
	TOURNAMENT_SIZE : int
	USE_RAND_MUT_TYPE : bool
	EPOCHS : int
	
	
	# fare in modo che in init vengano passati gli argmenti da linea di comando per decidere i vari iper parametri 
	# e le strategie come ad esempio quale tipo di mutazione usare.

	def __init__(self, 
			  problem : Problem, 
			  population_size : int = 100, 
			  offspring_size : int = 1_000, 
			  max_generations : int = 1_000,
			  mutation_type : MUTATION = MUTATION.POINT,
			  mutation_probability : float = 0.05,
			  tournament_size : int = 3,
			  use_random_mutation_type : bool = False,
			  epochs : int = 1_000) -> None:
		
		# initalize variables
		self.POPULATION_SIZE = population_size
		self.OFFSPRING_SIZE = offspring_size
		self.MAX_GENERATIONS = max_generations
		self.MUTATION_TYPE = mutation_type
		self.MUTATION_PROBABILITY = mutation_probability
		self.TOURNAMENT_SIZE = tournament_size
		self.USE_RAND_MUT_TYPE = use_random_mutation_type
		self.EPOCHS = epochs

		# extract problem data
		self.problem = problem
		self.use_validation = problem.use_validation_set
		self.mutation_type = mutation_type
		x = problem.x_train
		y = problem.y_train

		# init population
		for _ in range(self.POPULATION_SIZE):
			self.population.append(t.Tree(x, y))
	
	#tournament selection without replacement (try with replacement)
	def parent_selection(self):
		tournament_contestants = np.random.choice(self.population, self.TOURNAMENT_SIZE, replace=False) 
		best_candidate = max(tournament_contestants, key=lambda x: x.fitness) 
		return best_candidate
 	
	def mutation(self, 
			  individual : t.Tree, 
			  mut_type : MUTATION  = MUTATION.POINT,
			  use_random_mutation_type : bool = False) -> t.Tree:
		"""
		Select and apply a mutation to an individual

		:param individual:
		:type individual: t.Tree

		:param mut_type: The mutation used on the individual 
		:type mut_type: MUTATION(enum)
		W
		:param use_random_mutation_type: If true ignore the mut_type and select<br/>randomly a mutation type with eaven<br/>probability.
		:type use_random_mutation_type: bool
		"""

		if use_random_mutation_type:
			# select random mutation type
			mut_type = random.choice(list(self.MUTATION))
			
		# select mutation type
		match mut_type:
			case self.MUTATION.SUBTREE:
				individual = t.subtree_mutation(individual)
			case self.MUTATION.POINT:
				individual = t.point_mutation(individual)
			case self.MUTATION.PERMUT:
				individual = t.permutation_mutation(individual)
			case self.MUTATION.HOIST:
				individual = t.hoist_mutation(individual)
			case self.MUTATION.EXPANSION:
				individual = t.expansion_mutation(individual)
			case self.MUTATION.COLLAPSE:
				individual = t.collapse_mutation(individual)

		return individual

	def _step():
		pass
