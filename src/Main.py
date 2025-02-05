import Utils.ArgParser as ap
from Utils.ProblemLoader import ProblemList
from Utils.Saver import Saver

#!/usr/bin/python3
if __name__ == '__main__':
    opt = ap.parse_cmd_arguments()
    
    # load all the problems from path
    pl = ProblemList()
    pl.load_from_path(opt.dset, opt.count, opt.split, opt.ratio)

    # Save the problems solution in the s310582.py file
    saver = Saver(opt.out, opt.name, opt.id)
    for problem in pl.problems:
        saver.append_solution(problem)