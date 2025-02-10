# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import numbers
import numpy as np
from typing import Callable
from utils.arity import arity
import copy

class Node:
    _function: Callable
    _successors: list['Node']
    _parent: 'Node'
    _arity: int 
    _str: str 
    _leaf: bool 
    _type: str 

    def __init__(self, node=None, successors= None, *, name=None):
        
        self._parent = None
        
        #FIRST CASE: the node is a function 
        if callable(node):
            
            def _f (*_args, **_kwargs):
                try:
                    return node(*_args)
                except TypeError:
                    return np.nan
            
            self._function = _f
            self._type = 'f'
            self._arity = arity(node)
            self._successors = list(successors) if successors is not None else []
            
            self._leaf = False
            assert all(isinstance(s, Node) for s in self._successors), "Panic: Successors must be `Node`"
            
            for s in self._successors:
                s._parent = self
            
            if name is not None:
                self._str = name
            elif node.__name__ == '<lambda>':
                self._str = 'λ'
            else:
                self._str = "np." + node.__name__
                
        #SECOND CASE: the node is a number
        elif isinstance(node, numbers.Number):
            self._function = eval(f'lambda **_kw: {node}')
            self._successors = []
            self._arity = 0
            self._type = 'c'
            if name is not None:
                self._str = name
            else:
                self._str = f'{node:g}' 
            self._leaf = True
            
        #THIRD CASE: the node is a variable
        elif isinstance(node, str):
            self._function = eval(f'lambda *, {node}, **_kw: {node}')
            self._successors = []
            self._arity = 0
            self._type = 'v'
            self._str = str(node)
            self._leaf = True
        else:
            assert False
    
    #recursively calls the function (solves the expression tree)
    def __call__(self, **kwargs):
        res = self._function(*[c(**kwargs) for c in self._successors], **kwargs)
        
        #protection against dangerous operations
        if self._parent is not None and (self._parent.short_name == 'np.divide' or self._parent.short_name == 'np.log'):
            if(isinstance(res, np.ndarray)):
                res[res == 0] = 1e-6
            else:
                if res == 0:
                    res = 1e-6
        elif self._parent is not None and (self._parent.short_name == 'np.log' and self._parent.short_name == 'np.sqrt'):
            res[res < 0] = abs(res)
            if(isinstance(res, np.ndarray)):
                res[res == 0] = 1e-6
            else:
                if res == 0:
                    res = 1e-6
        return res
        
        
    def __str__(self):
        return self.long_name
    
    def __len__(self):
        return 1 + sum(len(s) for s in self._successors) 
    
    @property
    def is_leaf(self):
        return self._leaf
    
    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        if self._leaf is not False:
            return self.short_name
        else:
            return f'{self.short_name}(' + ', '.join(c.long_name for c in self._successors) + ')'
    
    def get_successors(self):
        return self._successors


    def __deepcopy__(self, memo):
        """Creates a deep copy of the current node."""
        if id(self) in memo:
            return memo[id(self)]
        
        # Creation of the new node based on the type
        if self._type == 'c': 
            copied_node = Node(float(self._str))
            return copied_node
        elif self._type == 'v': 
            copied_node = Node(self._str)
            return copied_node
        else:  
            copied_node = Node(eval(self.short_name), successors=[], name=self._str)
        
        # Recursively copy the successors
        copied_node._successors = [copy.deepcopy(s, memo) for s in self._successors]
        
        # Update the parent for the new successors
        for child in copied_node._successors:
            child._parent = copied_node

        # Save the new node in the memo
        memo[id(self)] = copied_node
        
        return copied_node