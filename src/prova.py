import numpy as np
import tree.tree as t
from tree.tree import Node


# rnd_tree = t.create_random_tree()
# print(str(rnd_tree))
# print(rnd_tree(x = 1, y = 3, z = 6))

problem = np.load('../data/problem_0.npz')
x = problem['x'] #variables
y = problem['y'] #results
print(x.shape, y.shape)

#Create the list of variables to be used in the tree
var = []

for i in range(x.shape[0]):
    var.append('x' + str(i))
    
print(var)
tree = t.Node('nan')

while str(tree) == 'nan':   
    tree = t.create_random_tree(var)


#voglio calcolare la fitness sulla singola entry del dataset o su tutto ul dataset?
def fitness(tree: t.Node, y: np.ndarray, x: np.ndarray) -> float:
    t = np.nan_to_num(tree(x0 = x[0], x1 = x[1]), nan = -10) #per ora i nan vengono sostituiti con un valore negativo (fitness peggiore)
    return - 100*np.square(y-t).sum()/len(y)

print(str(tree))
print(tree(x0 = x[0], x1 = x[1]))
print(fitness(tree, x, y))