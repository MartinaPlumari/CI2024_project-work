# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import functools
import tree.tree as t
from Utils.ProblemLoader import Problem
from enum import Enum


class Symreg:
	# data loading / initialization
	# counter 
	# cost (using the fitness function inside tree)

	problem : Problem
	# definire una classe quindi un tipo
	tree = None
	use_validation : bool

	class MUTATION(Enum):
		SUBTREE = 0
		POINT = 1
		PERMUT = 2
		HOIST = 3
		EXPANSION = 4
		COLLAPSE = 5
	
	mutation_type : MUTATION = MUTATION.SUBTREE
	POPULATION_SIZE : int = 100
	OFFSPRING_SIZE : int = 1_000
	MAX_GENERATIONS : int = 1_000
	GEN_OP_PROBABILITY : float = 0.4
	TOURNAMENT_SIZE : int = 3
	EPOCHS : int = 1000
	
	
	# fare in modo che in init vengano passati gli argmenti da linea di comando per decidere i vari iper parametri 
	# e le strategie come ad esempio quale tipo di mutazione usare.

	def __init__(self, problem : Problem, mutation_type : MUTATION = 0) -> None:
		self.problem = problem
		self.use_validation = problem.use_train_set
		self.mutation_type = mutation_type

		# tree initialization
		x = problem.x_validation
		y = problem.y_validation

		var = list()
		for i in range(x.shape[0]):
			var.append('x' + str(i))

		tree = t.create_random_tree(var)
	
	def _step():
		pass