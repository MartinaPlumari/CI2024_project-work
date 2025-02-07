import numpy as np
import tree.tree as t
from tree.node import Node
import random

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

t1 = t.Tree(x, y)
t2 = t.Tree(x, y)
#print("Albero da cui partire:")
#print(t1)
#print(t.get_tree_height(t1._root))
print("Albero prima della mutazione puntiforme:")
print(t1)
t1 = t.point_mutation(t1)
print("Albero dopo la mutazione:")
print(t1)

print("Albero prima della permutation mutation:")
print(t2)
t1 = t.permutation_mutation(t2)
print("Albero dopo la permutation mutation:")
print(t2)

# #prova funz insert e get_node
# n = list()
# n.append(random.randint(0, t2._n))
# nodo = t2.get_node(n)
# n2 = list()
# n2.append(random.randint(0, t1._n))
# print("Albero pre modifica:")
# print(t1)
# print(t1._n)
# print("Albero 2:")
# print(t2)
# print("Nodo da inserire:")
# print(nodo)
# t1.insert_node(n2, t1._root, nodo)
# print("Albero post modifica:")
# print(str(t1))
#print(nodo)
#print(t())
#print(t.fitness)

#tree = t.crossover(tree1, tree2)
#print(str)