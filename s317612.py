# CI 2024 final project
# Student None
# ID s317612

import numpy as np

def f0(x : np.ndarray) -> np.ndarray:
	return np.add(x[0], np.multiply(x[1], 0.180848))

def f1(x : np.ndarray) -> np.ndarray:
	return np.sin(x[0])

def f2(x : np.ndarray) -> np.ndarray:
	return np.multiply(5.31454, np.multiply(np.divide(x[0], np.subtract(np.tan(x[1]), np.tan(x[1]))), 2.54036))

def f3(x : np.ndarray) -> np.ndarray:
	return np.multiply(x[1], np.subtract(x[1], np.multiply(x[1], x[1])))

def f4(x : np.ndarray) -> np.ndarray:
	return np.subtract(2.79429, np.multiply(-7.08441, np.cos(x[1])))

def f5(x : np.ndarray) -> np.ndarray:
	return np.subtract(np.multiply(np.sqrt(x[0]), np.sqrt(x[0])), x[0])

def f6(x : np.ndarray) -> np.ndarray:
	return np.subtract(x[0], np.multiply(np.subtract(x[0], x[1]), np.sqrt(3.28835)))

def f7(x : np.ndarray) -> np.ndarray:
	return np.multiply(x[1], np.multiply(np.subtract(np.divide(np.cos(-8.4597), x[1]), x[0]), -7.52244))

def f8(x : np.ndarray) -> np.ndarray:
	return np.divide(x[5], np.subtract(x[3], x[3]))