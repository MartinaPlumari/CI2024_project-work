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
    parser.add_argument('-s', '--split', action='store_true', help='If used tells the loader to split the dataset into testing and validation.')
    parser.add_argument('-r', '--ratio', type=float, default=10, help='Ratio between training and testing dataset sizes.')
    parser.add_argument('-c', '--count', type=int, default=1, help='Number of problems to load for training.')
    parser.add_argument('-n', '--name', type=str, help='Student\'s name.')
    parser.add_argument('-i', '--id', type=str, help='Student\'s id like s123456.')
    return parser.parse_args()
