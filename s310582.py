# CI 2024 final project
# Student Daniel
# ID s310582

import numpy as np

def f0(x : np.ndarray) -> np.ndarray:
	return np.add(x[0], np.sin(np.subtract(x[0], x[0])))

