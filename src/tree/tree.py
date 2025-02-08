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

FUNCTIONS = [np.add, np.subtract, np.multiply, np.divide, np.tan, np.sin, np.cos, np.sqrt, np.log] #np.exp 
CONSTANT_RANGE = (-10, 10) #could be an eccessive limitation
MAX_DEPTH = 3
VARIABLE_P = 0.5
EARLY_STOP_P = 0.1

class init_method(Enum):
    GROW = 0
    FULL = 1

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
     
    def __init__(self, x: np.ndarray, y: np.ndarray, init_method: init_method = init_method.GROW, depth: int = MAX_DEPTH):   
        self._root = Node('nan')
        self._n = 0
        self._n_var = 0
        self._x = x
        self._y = y
        
        var = []
        self._n_var = x.shape[0]
        
        for i in range(self._n_var):
            var.append(f'x{str(i)}')
        
        self._root, self._n = create_random_tree(var, 0, depth, init_method)
        self._h = get_tree_height(self._root)
        self._fitness = self.fitness
        
    def __str__(self):
        return str(self._root)
    
    def __call__(self):
        kwargs = {f'x{i}': self._x[i] for i in range(self._n_var)}
        return self._root(**kwargs)
    
    @property
    def subtree(self):
         result = set()
         _get_subtree(result, self)
         return result   
        
    #cumulative fitness on all the dataset
    @property
    def fitness(self) -> float:
        kwargs = {f'x{i}': self._x[i] for i in range(self._n_var)}
        
        t = np.nan_to_num(self._root(**kwargs), nan = -10) #per ora i nan vengono sostituiti con un valore negativo (fitness peggiore)
        return - 100*np.square(self._y-t).sum()/len(self._y)
    
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
    
    def insert_node(self, n: list, parent: Node, ins_node: Node):
        
        if n[0] == 0:
            if parent.is_leaf:
                n[0] += 1
                return
            
            if parent._successors:
                print("Modifica del nodo...")
                idx = rnd.randint(0, len(parent._successors)-1)
                parent._successors[idx] = ins_node
                
            return 
        

        for s in parent._successors:
            n[0] -= 1
            if n[0] < 0:
                break 
            self.insert_node(n, s, ins_node)   
        
        return 
    
    

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
def create_random_tree(vars, depth = 0, max_depth = MAX_DEPTH, mode = init_method.GROW) -> tuple[Node, int]:
    """Recursively creates a random syntax tree"""
    
    node_count = 1
    
    # Base case: If max depth is reached, return a leaf node (constant or variable)
    #if grow is selected as a mode, we have a chance of stopping early
    if depth >= max_depth or (mode == init_method.GROW and rnd.random() < EARLY_STOP_P):  
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
    
        # Handle edge cases
    if func == np.log:  # Ensure log input is positive
        successors = [Node(np.absolute, successors)]
        node_count += 1  # Add new absolute node
                   
    elif func == np.divide:  # Ensure denominator is not zero
        if successors[1]._leaf and successors[1]._type == 'c' and -0.001 < successors[1]() < 0.001:
            successors[1] = Node(1)  # Replace zero denominator

    elif func == np.sqrt:  # Ensure sqrt input is positive
        successors = [Node(np.absolute, successors)]
        node_count += 1  # Add new absolute node
    
    return Node(func, successors), node_count


def crossover(t1: Tree, t2: Tree) -> Tree:
    """ Performs a crossover operation between two trees by taking a subtree from the second one and replacing a subtree in the first one."""
    root1 = t1._root

    n1 = rnd.randint(0, t1._n-1)
    n2 = rnd.randint(0, t2._n-1)
    
    node = t2.get_node([n2])
    t1.insert_node([n1], root1, node)
    t1._n = count_nodes(t1._root)
    t1._h = get_tree_height(t1._root)
    
    return t1

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
    
    #this has debug purposes: to be deleted
    #print(f"Replacing function {node._str} with {func.__name__}")
    
    if func == np.log and node._successors[0].short_name != 'np.absolute':
         node._successors = [Node(np.absolute, node._successors)]
    elif func == np.divide:
        if node._successors[1]._leaf and node._successors[1]._type == 'c' and -0.001 < node._successors[1]() < 0.001:
            node._successors[1] = Node(1)
    elif func == np.sqrt:
        if node._successors[0]._leaf and node._successors[0]._type == 'c' and -0.001 < node._successors[0]() < 0.001:
            node._successors[0] = Node(0.1)
        if node._successors[0].short_name != 'np.absolute':
            node._successors = [Node(np.absolute, node._successors)]
    
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
    
    valid_nodes = [i for i in range(t._n) if not t.get_node([i]).is_leaf and t.get_node([i])._arity > 1]
    
    if not valid_nodes:
        return t
    
    n1 = rnd.choice(valid_nodes)
    node1 = t.get_node([n1])
    
    node1._successors = node1._successors[::-1]
    
    return t

def hoist_mutation(t: Tree) -> Tree:
    """Performs a hoist mutation by replacing the root node with a random subtree."""
    
    if t._n < 3:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf:
        n = rnd.randint(1, t._n-1)
        node = t.get_node([n])
    
    t._root = node
    t._h = get_tree_height(t._root)
    t._n = count_nodes(t._root)
    
    return t

def collapse_mutation(t: Tree) -> Tree:
    """Performs a collapse mutation by replacing a subtree with a leaf node."""
    
    if t._n < 4:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf or node._parent is None or node.short_name == 'np.absolute':
        n = rnd.randint(1, t._n-1)
        node = t.get_node([n])
    
    var = []
    for i in range(t._n_var):
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
    
    if t._n < 3:
        return t
    
    n = rnd.randint(1, t._n-1)
    node = t.get_node([n])
    
    while node is None or node.is_leaf or node.short_name == 'np.absolute':
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
    for i in range(t._n_var):
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
    return t