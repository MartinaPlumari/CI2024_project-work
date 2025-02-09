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

    problem : Problem = pl.problems[1]

    alg : Symreg = Symreg(problem, 300, 300, 1000, Symreg.MUTATION.EXPANSION, Symreg.POPULTAION_MODEL.STEADY_STATE, 0.01, 3, False)
    alg.train()

    print(f"RESULT: {alg.problem.solution._root}\nFITNESS: {alg.problem.solution._fitness}")

    saver.append_solution(alg.problem)

    alg.plot_history()
    draw.draw_tree(alg.problem.solution._root)