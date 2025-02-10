# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import argparse

logo : str = """
  ____ ___     ____   ___ ____  _  _   
 / ___|_ _|   |___ \ / _ \___ \| || |  
| |    | |_____ __) | | | |__) | || |_ 
| |___ | |_____/ __/| |_| / __/|__   _|
 \____|___|   |_____|\___/_____|  |_|  
"""

epilogue : str = """
This code is licensed under the MIT License. Copyright (c) 2025 Martina Plumari, Daniel Bologna.
Developed for the course "Computational Intelligence" at Politecnico di Torino.
"""

def parse_cmd_arguments():
    """
    Parses the main configuration parameters from the commandline.
    """
    parser = argparse.ArgumentParser(
                    prog='../Main.py',
                    description=logo,
                    epilog=epilogue,
                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dset', type=str, default="../data/", help='Dataset path.')
    parser.add_argument('-o', '--out', type=str, default="../", help='Output path.')
    # parser.add_argument('-s', '--split', action='store_true', help='If used tells the loader to split the dataset into testing and validation.')
    # parser.add_argument('-r', '--ratio', type=float, default=10, help='Ratio between training and testing dataset sizes.')
    parser.add_argument('-c', '--count', type=int, default=1, help='Number of problems to load for training.')
    parser.add_argument('-i', '--index', type=int, default=0, help='Index of the problem to use for training.')
    parser.add_argument('-a', '--all', action='store_true', help='If used ignore the problem index and start training with all the problems sequentially.')
    parser.add_argument('--popsize', type=int, default=100, help='Population size.')
    parser.add_argument('--offsize', type=int, default=100, help='Offspring size.')
    parser.add_argument('--maxgen', type=int, default=100, help='Max train generations.')
    parser.add_argument('--mtype', type=int, default=0, help="""Type of applied mutation:\n\t- SUBTREE = 0\n\t- POINT = 1\n\t- PERMUT = 2\n\t- HOIST = 3\n\t- EXPANSION = 4\n\t- COLLAPSE = 5""")
    parser.add_argument('--pmodel', type=int, default=0, help="""Population model:\n\t-STEADY_STATE = 0\n\t-GENERATIONAL = 1""")
    parser.add_argument('--pinitmodel', type=int, default=1, help="""Population initialization model:\n\t-GROW = 0 // trees have chance of being shorter than max depth\n\t-FULL = 1 // All trees reach max depth\n\t-HALF_HALF = 2 // half of population is GROW and other half is FULL""")
    parser.add_argument('--mutchance', type=float, default=0.05, help='Initial probability to mutate an individual instead of performing recombination.')
    parser.add_argument('--tsize', type=int, default=3, help='Max number of individuals for tournament style parent selection.')
    parser.add_argument('--random', action='store_true', help='If enabled, during training select a random mutation instead of using mtype.')
    parser.add_argument('--name', type=str, default='Name Surname', help='Student\'s name.')
    parser.add_argument('--id', type=str, default='s123456', help='Student\'s id like s123456.')
    return parser.parse_args()
