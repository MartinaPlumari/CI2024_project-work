# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import numpy as np
import random as rnd
from enum import Enum 
from tree.node import Node
from utils.arity import arity
import copy

FUNCTIONS = [np.add, np.subtract, np.multiply, np.divide, np.tan, np.sin, np.cos, np.sqrt, np.log]#, np.power] #np.exp
CONSTANT_RANGE = (-10, 10) #could be an eccessive limitation
MAX_DEPTH = 4
VARIABLE_P = 0.7
EARLY_STOP_P = 0.0

class Tree:
    """
    TODO
    """
    
    _root: Node
    _n: int
    _h: int
    _x: np.ndarray
    _y: np.ndarray
    _fitness: float
     
    def __init__(self, x: np.ndarray, y: np.ndarray, INIT_METHOD: int = 0, depth: int = MAX_DEPTH):   
        self._root = Node('nan')
        self._n = 0
        self._x = x
        self._y = y
        self._kwargs : dict = {f'x{i}': self._x[i] for i in range(self._x.shape[0])}
        
        var = []
        
        for i in range(self._x.shape[0]):
            var.append(f'x{str(i)}')
        
        self._root, self._n = create_random_tree(var, 0, depth, INIT_METHOD)
        self._h = get_tree_height(self._root)
        self._fitness = self.fitness
        
    def __str__(self):
        return str(self._root)
    
    def __call__(self):
        return self._root(**self._kwargs)
    
    @property
    def subtree(self):
         result = set()
         _get_subtree(result, self)
         return result   
        
    #cumulative fitness on all the dataset
    @property
    def fitness(self) -> float:        
        t = np.nan_to_num(self._root(**self._kwargs), nan = -10)
        return - np.mean((t - self._y) ** 2)
    
    def get_node(self, n: list, node: Node = None) -> Node:
        if node is None:
            node = self._root
        
        if n[0] == 0:
            return node
        
        n[0] -= 1
        
        for s in node._successors:
            res = self.get_node(n, s)
            if res is not None:
                return res
        
        return None
    
    def deep_copy(self) -> 'Tree':
        new_tree = Tree(self._x, self._y)
        new_tree._root = copy.deepcopy(self._root)
        new_tree._n = self._n
        new_tree._h = self._h
        new_tree._fitness = self._fitness
        new_tree._kwargs = self._kwargs
        return new_tree
       
    
def _get_subtree(bunch: set, node: Node):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c) 
    
def get_tree_height(node: Node) -> int: 
    """Recursively calculates the height of a given tree."""
    if node.is_leaf:
        return 1  # Leaf nodes have height 1
    
    return 1 + max(get_tree_height(child) for child in node.get_successors())

def count_nodes(node):
    """Recursively counts the total number of nodes in the tree."""
    return 1 + sum(count_nodes(child) for child in node.get_successors())

#problema, spesso escono valori nan o errori (divide by 0, log di numeri negativi, ecc)
def create_random_tree(vars, depth = 0, max_depth = MAX_DEPTH, mode: int = 0) -> tuple[Node, int]:
    """Recursively creates a random syntax tree"""
    
    node_count = 1
    
    # Base case: If max depth is reached, return a leaf node (constant or variable)
    #if grow is selected as a mode, we have a chance of stopping early
    if depth >= max_depth or (mode == 0 and rnd.random() < EARLY_STOP_P):  
        # Random variable
        if rnd.random() < VARIABLE_P:
            return Node(rnd.choice(vars)), node_count  
        # Random constant
        else:
            return Node(rnd.uniform(*CONSTANT_RANGE)), node_count  
    
    # Recursive case: Create a function node
    func = rnd.choice(FUNCTIONS)  # Choose a random function/operator
    ar = arity(func)  # Get function arity
    
    successors = []
    for _ in range(ar):
        child, child_count = create_random_tree(vars, depth + 1, max_depth)
        successors.append(child)
        node_count += child_count  # Accumulate total nodes created
    
    return Node(func, successors), node_count

def recombination(t1: Tree, t2: Tree) -> Tree:
    if t1._n < 2 or t2._n < 2:
        return
    
    # Select a random node from t1 (excluding the root)
    n1 : Node = rnd.randint(1, t1._n - 1)  
    node1 = t1.get_node([n1])
    
    # get random node from t1
    while node1 is None or node1._parent is None:
        n1 = rnd.randint(1, t1._n - 1)
        node1 = t1.get_node([n1])
    
    n2 : Node = rnd.randint(1, t2._n-1)
    node2 = t2.get_node([n2])
    
    #get random node from t2
    while node2 is None:
        n2 = rnd.randint(1, t2._n - 1)
        node2 = t2.get_node([n2])
    
    node1._parent._successors[node1._parent._successors.index(node1)] = node2
    node2._parent._successors[node2._parent._successors.index(node2)] = node1
    
    tmp : Node = node1._parent
    node1._parent = node2._parent
    node2._parent = tmp

    t1._fitness = t1.fitness 
    t2._fitness = t2.fitness
    t1._h = get_tree_height(t1._root)
    t1._n = count_nodes(t1._root)
    t2._h = get_tree_height(t2._root)
    t2._n = count_nodes(t2._root)

    return t1

    

def crossover(t1: Tree, t2: Tree) -> Tree:
    """ Performs a crossover operation between two trees by taking a subtree from the second one and replacing a subtree in the first one."""
    t = t1.deep_copy()
    
    if t1._n < 2:
        return t

    # Select a random node from t1 (excluding the root)
    n1 = rnd.randint(1, t._n - 1)  
    node1 = t.get_node([n1])
    
    while node1 is None or node1._parent is None:
        n1 = rnd.randint(1, t._n - 1)
        node1 = t.get_node([n1])
    
    n2 = rnd.randint(0, t2._n-1)
    node2 = t2.get_node([n2])
    
    while node2 is None:
        n2 = rnd.randint(0, t2._n - 1)
        node2 = t2.get_node([n2])
    
    new_subtree = copy.deepcopy(node2)
    parent = node1._parent
    
    if parent is not None:
        new_subtree._parent = parent
        parent._successors[parent._successors.index(node1)] = new_subtree
    else:
        t = t1.deep_copy()
        t._root = new_subtree
    
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    
    return t

def point_mutation(t: Tree) -> Tree:
    """Performs a mutation by replacing a function node with a different random function."""
    
    if t._n < 2:
        return t
    
    n = rnd.randint(0, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf or node.short_name == 'np.absolute':
        n = rnd.randint(0, t._n-1)
        node = t.get_node([n])
    
    old_arity = node._arity
    func = rnd.choice([f for f in FUNCTIONS if (("np." + f.__name__) != node.short_name and arity(f) == old_arity)])           
    
    new_node = Node(func, node._successors)
    parent = node._parent
    if(parent is None):
        t._root = new_node
    else:
        new_node._parent = parent
        node._parent._successors[parent._successors.index(node)] = new_node  
    
    return t

def permutation_mutation(t: Tree) -> Tree:
    """Exchanges two lives in the tree"""
    
    if t._n < 3:
        return t
    
    valid_nodes = [i for i in range(t._n-1) if  t.get_node([i]) is not None and not t.get_node([i]).is_leaf and t.get_node([i])._arity > 1]
    
    if len(valid_nodes) == 0:
        return t
    
    n1 = rnd.choice(valid_nodes)
    node1 = t.get_node([n1])
    
    node1._successors = node1._successors[::-1]
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def hoist_mutation(t: Tree) -> Tree:
    """Performs a hoist mutation by replacing the root node with a random subtree."""
    
    if t._n < 4:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf:
        n = rnd.randint(1, t._n-1)
        node = t.get_node([n])
    
    t._root = node
    t._root._parent = None
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def collapse_mutation(t: Tree) -> Tree:
    """Performs a collapse mutation by replacing a subtree with a leaf node."""
    
    if t._n < 4:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf or node._parent is None:# or node.short_name == 'np.absolute':
        n = rnd.randint(1, t._n-1)
        node = t.get_node([n])
    
    var = []
    for i in range(t._x.shape[0]):
        var.append(f'x{str(i)}')
    
    if rnd.random() < VARIABLE_P:
        new_node = Node(rnd.choice(var)) 
        # Random constant
    else:
        new_node = Node(rnd.uniform(*CONSTANT_RANGE))
        
    parent = node._parent
    new_node._parent = parent
    parent._successors[parent._successors.index(node)] = new_node
    
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def subtree_mutation(t: Tree) -> Tree:
    """Performs a subtree mutation by replacing a subtree with a one of its subtree."""
    
    if t._n < 4:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf: # or node.short_name == 'np.absolute':
        n = rnd.randint(1, t._n-1)
        node = t.get_node([n])
    
    new_node = node._successors[rnd.randint(0, len(node._successors)-1)]
    parent = node._parent
    new_node._parent = parent
    parent._successors[parent._successors.index(node)] = new_node
    
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def expansion_mutation(t: Tree) -> Tree:
    """Performs an expansion mutation by replacing a leaf node with a random subtree."""
    
    n = rnd.randint(0, t._n-1)
    node = t.get_node([n])
    
    while node is None or not node.is_leaf:
        n = rnd.randint(0, t._n-1)
        node = t.get_node([n])
    
    var = []
    for i in range(t._x.shape[0]):
        var.append(f'x{str(i)}')
    
    new_node, _ = create_random_tree(var, 0, rnd.randint(1, MAX_DEPTH))
    
    parent = node._parent
    new_node._parent = parent
    parent._successors[parent._successors.index(node)] = new_node
    
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def simplify_tree(t: Tree) -> Tree:
    """Simplifies the tree by evaluating and replacing constant expressions."""
    
    def simplify_node(node: Node):
        """Recursively simplifies a node."""
        if node.is_leaf:
            return node  
        
        # Simplify all successors first (post-order traversal)
        for i, child in enumerate(node._successors):
            node._successors[i] = simplify_node(child)
        
        # If all successors are constants, evaluate the node
        if all(child.is_leaf and child._type == 'c' for child in node._successors):
            try:
                new_value = node()  
                return Node(new_value)  
            except Exception:
                return node  
        
        return node  

    t._root = simplify_node(t._root) 
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t