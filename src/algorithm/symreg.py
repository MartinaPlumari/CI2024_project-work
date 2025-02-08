# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import tree.tree as t
from utils.problemloader import Problem
from enum import Enum


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
			  epochs : int = 1_000) -> None:
		
		# initalize variables
		self.POPULATION_SIZE = population_size
		self.OFFSPRING_SIZE = offspring_size
		self.MAX_GENERATIONS = max_generations
		self.MUTATION_TYPE = mutation_type
		self.MUTATION_PROBABILITY = mutation_probability
		self.TOURNAMENT_SIZE = tournament_size
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
		
	def mutation(self, individual : t.Tree, mut_type : MUTATION):
		match mut_type:
			case self.MUTATION.SUBTREE:
				pass
			case self.MUTATION.POINT:
				pass
			case self.MUTATION.PERMUT:
				pass
			case self.MUTATION.HOIST:
				pass
			case self.MUTATION.EXPANSION:
				pass
			case self.MUTATION.COLLAPSE:
				pass
	
	def _step():
		pass
