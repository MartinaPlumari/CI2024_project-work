# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import os
import numpy as np

class Problem:
    """Problem container for easly handle input data."""
    problem_id : int

    use_validation_set : bool
    
    train_size : int
    valid_size : int
    
    x_train : np.ndarray
    y_train : np.ndarray
    x_validation : np.ndarray
    y_validation : np.ndarray

    solution : str

    def __init__(self, path : str, id : int = -1, split : bool = False, ratio : int = 10):
        self.problem_id = id
        self.solution = ""

        # load dataset
        data = np.load(path)

        self.x_train = data['x']
        self.y_train = data['y']
        
        self.train_size = self.y_train.shape[0]
        self.use_validation_set = split

        assert self.train_size > 0, "The problem is empty, please load a valid problem file."

        log = f"Problem N.{self.problem_id} loaded successfully --\nTRAIN SIZE:\t\t{self.train_size}"

        if self.use_validation_set:
            # set the train_size as a % of the validation size
            self.valid_size = round(self.train_size / 100 * ratio)
            valid_indices = np.random.choice(self.train_size, size=self.valid_size, replace=False)
            self.x_valid = self.x_train[:, valid_indices]
            self.y_valid = self.y_train[valid_indices]
            log += f"\nVALIDATION SIZE:\t{self.valid_size}\n"

        print(log)

class ProblemList:
    path : str
    problems : list[Problem]

    def __init__(self) -> None:
        """Constructor"""
        self.problems = list()
    
    def load_from_path(self, path : str, count : int, split : bool = False, ratio : int = 10) -> None:
        """Load all problems from specified directory.
        
        Parameters
        ---
        path : `str`
            Dataset folder path.
        split : `bool`
            Wether to split or not the dataset in training and validation.
        ratio : `int`
            Percentage of the dataset to allocate for training. The rest will be used for validation at the end.
        """
        
        assert ratio >= 0 and ratio <= 100, "Ratio must be defined in [0, 100] range."
        assert count > 0, "Count can't be negative."

        self.path = path

        files = os.listdir(self.path)
        
        if count >= len(files):
            print(f"\nTried to load: {count} problems but only {len(files)} have been found.\n")
            count = len(files)

        for i in range(count):
            if files[i].endswith(".npz"):
                file_path = path + '/' + files[i]
                self.problems.append(Problem(file_path, i, split, ratio))
        
        if len(self.problems) <= 0:
            raise FileNotFoundError("No data found in given directory: \"" + self.path + "/\" -- Please select a valid folder path.")