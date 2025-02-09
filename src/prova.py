import numpy as np
import tree.tree as t
from tree.node import Node
import copy

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
print(str(t1))
#print("Albero da cui partire:")
#print(t1)
#print(t.get_tree_height(t1._root))
# print("Albero prima della mutazione puntiforme:")
# print(t1)
# t1 = t.point_mutation(t1)
# print("Albero dopo la mutazione:")
# print(t1)

# print("Albero prima della permutation mutation:")
# print(t2)
# t2 = t.permutation_mutation(t2)
# print("Albero dopo la permutation mutation:")
# print(t2)

# print("Albero prima della semplificazione:")
# print(t2)
# t2 = t.simplify_tree(t2)
# print("Albero dopo la semplificazione:")
# print(t2)

# print("Albero prima della hoist mutation:")
# print(t1)
# t1 = t.hoist_mutation(t1)
# print("Albero dopo la hoist mutation:")
# print(t1)

# print("Albero prima della collapse mutation:")
# print(t1)
# t1 = t.collapse_mutation(t1)
# print("Albero dopo la collapse mutation:")
# print(t1)

# print("Albero prima della expansion mutation:")
# print(t1)
# t1 = t.expansion_mutation(t1)
# print("Albero dopo la expansion mutation:")
# print(t1)

# print("Albero prima della subtree mutation:")
# print(t1)
# t1 = t.subtree_mutation(t1)
# print("Albero dopo la subtree mutation:")
# print(t1)

# print("Albero 1 prima del crossover:")
# print(t1)
# print("Albero 2 prima del crossover:")
# print(t2)
# t1 = t.crossover(t1, t2)
# print("Albero dopo il crossover:")
# print(t1)


#print(t2)
print(t1)

# tn = t1.deep_copy()
# print(tn)
# t1 = t.subtree_mutation(t1)
# print(t1)
# print(tn)
# print(tn._n)
# print(t1._fitness)
print(t1())

# for i in range(10):
#     blallo = t.crossover(t1, tn)
#     print(blallo)
#     print(blallo._n)