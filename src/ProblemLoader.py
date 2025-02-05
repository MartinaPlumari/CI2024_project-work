import os
import numpy as np

class Problem:
    """Problem container for easly handle input data."""
    problem_id : int

    use_train_set : bool
    
    valid_size : int
    train_size : int
    
    x_validation : np.ndarray
    y_validation : np.ndarray
    x_train : np.ndarray
    y_train : np.ndarray

    solution : str

    def __init__(self, path : str, id : int = -1, split : bool = False, ratio : int = 10):
        self.problem_id = id
        self.solution = ""

        # load dataset
        data = np.load(path)

        self.x_validation = data['x']
        self.y_validation = data['y']
        
        self.valid_size = self.y_validation.shape[0]
        self.use_train_set = split

        assert self.valid_size > 0, "The problem is empty, please load a valid problem file."

        log = f"Problem N.{self.problem_id} loaded successfully --\nVALIDATION SIZE:\t{self.valid_size}"

        if self.use_train_set:
            # set the train_size as a % of the validation size
            self.train_size = round(self.valid_size / 100 * ratio)
            train_indexes = np.random.choice(self.valid_size, size=self.train_size, replace=False)
            self.x_train = self.x_validation[:, train_indexes]
            self.y_train = self.y_validation[train_indexes]
            log += f"\nTRAIN SIZE:\t\t{self.train_size}"

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