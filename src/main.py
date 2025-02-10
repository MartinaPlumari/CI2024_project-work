# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

from tree.tree import Tree

import utils.argparser as ap
from utils.problemloader import ProblemList, Problem
from utils.saver import Saver
import utils.draw as draw
from algorithm.symreg import Symreg


#!/usr/bin/python3
if __name__ == '__main__':
    opt = ap.parse_cmd_arguments()
        
    # load all the problems from path
    pl = ProblemList()
    pl.load_from_path(opt.dset, opt.count, opt.split, opt.ratio)

    # save the problems solution in the s310582.py file
    saver = Saver(opt.out, opt.name, opt.id)

    problems : list[Problem] = [pl.problems[i] for i in range(len(pl.problems)) if opt.all or i == opt.index]
    print([p.problem_id for p in problems])
    for i in range(len(problems)):
        print(f"\nStarting problem {problems[i].problem_id}")
        alg : Symreg = Symreg(problem=problems[i], 
                            population_size=opt.popsize, 
                            offspring_size=opt.offsize, 
                            max_generations=opt.maxgen, 
                            mutation_type=opt.mtype, 
                            population_model=opt.pmodel,
                            mutation_probability=opt.mutchance,
                            tournament_size=opt.tsize,
                            use_random_mutation_type=opt.random,
                            population_init_method=opt.pinitmodel)
        alg.train()
        print(f"\nRESULT: {alg.problem.solution._root}\nFITNESS: {alg.problem.solution._fitness}\n================================\n")
        saver.append_solution(alg.problem)

    # alg.plot_history()
    # draw.draw_tree(alg.problem.solution._root)