import sys
import numbers
import numpy as np
import random as rnd
from .node import Node
from .utils import arity

FUNCTIONS = [np.add, np.subtract, np.multiply, np.divide, np.tan, np.sin, np.cos, np.exp, np.sqrt, np.log] 
CONSTANT_RANGE = (-10, 10) #could be an eccessive limitation
MAX_DEPTH = 3

class Tree:
    _root: Node
    _n: int
    _h: int
    _x: np.ndarray #capire se ha senso mettere le variabili come attributi della classe
    _y: np.ndarray
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._root = Node('nan')
        self._n = 0
        self._n_var = 0
        self._x = x
        self._y = y
        
        var = []
        self._n_var = x.shape[0]
        
        for i in range(self._n_var):
            var.append('x' + str(i))
        
        self._root = create_random_tree(var)
        
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
    
    #capire se modificare la struttura dati per evitare la tupla dei successori (perchè non usare una lista?)
    #attraversa nel modo giusto ma non inserisce correttamente il nodo
    def insert_node(self, n: list, parent: Node, ins_node: Node, index: int = 0):
        if n[0] == 0:
            if parent.is_leaf:
                n[0] += 1
                return
            print("popi")
            print(parent)
            successors_list = list(parent._successors) 
            successors_list[index] = ins_node
            parent._successors = tuple(successors_list)
            print(parent)
            return 
        
        n[0] -= 1
        
        for i,s in enumerate(parent._successors):
            #print(i)
            #print(s)
            self.insert_node(n, s, ins_node, index = i)    
        
        return 
    

def _get_subtree(bunch: set, node: Node):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c) 
    

#problema, spesso escono valori nan o errori (divide by 0, log di numeri negativi, ecc)
def create_random_tree(vars, depth = 0, max_depth = MAX_DEPTH):
    """Recursively creates a random syntax tree."""
    
    # Base case: If max depth is reached, return a leaf node (constant or variable)
    if depth >= max_depth or rnd.random() < 0.1:  # 10% chance to stop early
        if rnd.random() < 0.5:
            return Node(rnd.choice(vars))  # Random variable
        else:
            return Node(rnd.uniform(*CONSTANT_RANGE))  # Random constant
    
    # Recursive case: Create a function node
    func = rnd.choice(FUNCTIONS)  # Choose a random function/operator
    ar = arity(func)  # Get function arity
    
    if func == np.log:  # Ensure log input is positive
        successors = [create_random_tree(vars, depth + 1, max_depth)]
        if successors[0]._leaf is True and successors[0]._type == 'c':
            if successors[0]() > 0 and successors[0]() < 0.001:
                successors[0] = Node(0)
        successors= [Node(np.absolute, successors)]  # Replace with a safe number
                   
    elif func == np.divide:  # Ensure denominator is not zero
        num = create_random_tree(vars, depth + 1, max_depth)
        denom = create_random_tree(vars, depth + 1, max_depth)
        if denom._leaf is True and denom._type == 'c':
            if denom() > -0.001 and denom() < 0.001:
                denom = Node(1)  # Replace zero denominator

        successors = [num, denom]
        
    elif func == np.sqrt:  # Ensure log input is positive
        successors = [create_random_tree(vars, depth + 1, max_depth)]
        successors = [Node(np.absolute, successors)]

    else:
        # Generate child nodes recursively
        successors = [create_random_tree(vars, depth + 1, max_depth) for _ in range(ar)]
    
    return Node(func, successors)

#DEBUGGA STA ROBA MI RACCOMANDO CHE NON FUNZIONA
def crossover(t1: Node, t2: Node) -> Node:
    """ Performs a crossover operation between two trees by taking a subtree from the second one and replacing a subtree in the first one."""
    ...