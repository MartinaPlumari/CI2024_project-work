import numpy as np
import tree.tree as t
from tree.node import Node

problem = np.load('../data/problem_0.npz')
x = problem['x'] #variables
y = problem['y'] #results
print(x.shape, y.shape)

#Create the list of variables to be used in the tree
var = []

for i in range(x.shape[0]):
    var.append('x' + str(i))
    
print(var)
tree1 = t.Node('nan')
tree2 = t.Node('nan')

#forse queste non servono: fare check
#while str(tree1) == 'nan':   
tree1 = t.create_random_tree(var)
    
#while str(tree2) == 'nan':   
tree2 = t.create_random_tree(var)

t = t.Tree(x, y)
n = [3]
nodo = t.get_node(n)
n2 = [6]
print(t)
print(nodo)
t.insert_node(n2, t._root, nodo)
#print(str(t))
#print(nodo)
#print(t())
#print(t.fitness)

#tree = t.crossover(tree1, tree2)
#print(str)