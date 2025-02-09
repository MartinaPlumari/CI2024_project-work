# CI 2024 final project
# Student Daniel
# ID s310582

import numpy as np

def f3(x : np.ndarray) -> np.ndarray:
	return np.multiply(x[1], np.subtract(x[0], np.multiply(x[1], x[1])))

