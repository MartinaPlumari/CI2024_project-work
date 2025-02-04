import numbers
import numpy as np
import random as rnd
from typing import Callable
from .utils import arity

#questo codice prbabilmente è overkill nel caso in cui vogliamo schematizzare solo espressioni matematiche
#obiettivo: da x+cos(b+y)+a ad add(x,cos(add(b,y)),a) o add(add(x,cos(add(b,y)),a)) a struttura ad albero e poi viceversa

class Node:
    _function: Callable
    _successors: tuple['Node'] #valutare se lasciare una tupla
    _arity: int #questa mi serve per capire come strutturare i successori
    _str: str #penso sia una sorta di etichetta (stringa) per il nodo
    _leaf: bool #questo mi serve per capire se il nodo è una foglia


    def __init__(self, node=None, successors= None, *, name=None):
        #FIRST CASE: the node is a function (or an operator, the two cases can coincide when using np)
        if callable(node):
            
            def _f (*_args, **_kwargs):
                try:
                    return node(*_args)
                except TypeError:
                    return np.nan
            
            self._function = _f
            self._successors = tuple(successors)
            self._arity = arity(node)
        
            assert self._arity is None or len(tuple(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(tuple(successors))} found {arity(node)}"
            )
            
            self._leaf = False
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"
            self._successors = tuple(successors)
            if name is not None:
                self._str = name
            elif node.__name__ == '<lambda>':
                self._str = 'λ'
            else:
                self._str = "np." + node.__name__
                
        #SECOND CASE: the node is a number
        elif isinstance(node, numbers.Number):
            self._function = eval(f'lambda **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            if name is not None:
                self._str = name
            else:
                self._str = f'{node:g}' 
            self._leaf = True
        #THIRD CASE: the node is a variable
        elif isinstance(node, str):
            self._function = eval(f'lambda *, {node}, **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            self._str = str(node)
            self._leaf = True
        else:
            assert False
    
    def __call__(self, **kwargs):
        return self._function(*[c(**kwargs) for c in self._successors], **kwargs)
            
    def __str__(self):
        return self.long_name
    
    def __len__(self):
        return 1 + sum(len(s) for s in self._successors) 

    
    def is_leaf(self):
        return self._leaf
    
    def get_successors(self):
        return self._successors
    
    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        if self._leaf is not False:
            return self.short_name
        else:
            return f'{self.short_name}(' + ', '.join(c.long_name for c in self._successors) + ')'


FUNCTIONS = [np.add, np.subtract, np.multiply, np.divide, np.power, np.tan, np.sin, np.cos, np.exp, np.sqrt, np.log] 
CONSTANT_RANGE = (-10, 10) #could be an eccessive limitation
MAX_DEPTH = 3

#problema, spesso escono valori nan o errori (divide by 0, log di numeri negativi, ecc)
def create_random_tree(vars, depth = 0, max_depth = MAX_DEPTH):
    """Recursively creates a random syntax tree."""
    
    # Base case: If max depth is reached, return a leaf node (constant or variable)
    if depth >= max_depth or rnd.random() < 0.1:  # 30% chance to stop early
        if rnd.random() < 0.8:
            return Node(rnd.choice(vars))  # Random variable
        else:
            return Node(rnd.uniform(*CONSTANT_RANGE))  # Random constant
    
    # Recursive case: Create a function node
    func = rnd.choice(FUNCTIONS)  # Choose a random function/operator
    ar = arity(func)  # Get function arity
    
    if func == np.log:  # Ensure log input is positive
        successors = [create_random_tree(vars, depth + 1, max_depth)]
        if isinstance(successors[0], Node) and successors[0]._leaf is True and isinstance(successors[0], numbers.Number):
            if successors[0] <= 0:
                successors[0] = Node(rnd.uniform(1, 10))  # Replace with a safe number
                
    elif func == np.divide:  # Ensure denominator is not zero
        num = create_random_tree(vars, depth + 1, max_depth)
        denom = create_random_tree(vars, depth + 1, max_depth)
        if isinstance(denom, Node) and denom._leaf is True and isinstance(denom, numbers.Number):
            if denom == 0:
                denom = Node(rnd.uniform(1, 10))  # Replace zero denominator

        successors = [num, denom]
        
    elif func == np.sqrt:  # Ensure log input is positive
        successors = [create_random_tree(vars, depth + 1, max_depth)]
        if isinstance(successors[0], Node) and successors[0]._leaf is True and isinstance(successors[0], numbers.Number):
            if successors[0] < 0:
                successors[0] = Node(rnd.uniform(0, 10))  # Replace with a safe number    
                            
    else:
        # Generate child nodes recursively
        successors = [create_random_tree(vars, depth + 1, max_depth) for _ in range(ar)]
    
    return Node(func, successors)
