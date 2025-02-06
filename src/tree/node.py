import numbers
import numpy as np
import random as rnd
from typing import Callable
from .utils import arity

class Node:
    _function: Callable
    _successors: tuple['Node'] #valutare se lasciare una tupla
    _arity: int #questa mi serve per capire come strutturare i successori
    _str: str #penso sia una sorta di etichetta (stringa) per il nodo
    _leaf: bool #questo mi serve per capire se il nodo è una foglia
    _type: str #questo mi serve per capire se il nodo è una variabile, una costante o una funzione


    def __init__(self, node=None, successors= None, *, name=None):
        #FIRST CASE: the node is a function (or an operator, the two cases can coincide when using np)
        if callable(node):
            
            def _f (*_args, **_kwargs):
                try:
                    return node(*_args)
                except TypeError:
                    return np.nan
            
            self._function = _f
            self._type = 'f'
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
            self._type = 'c'
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
            self._type = 'v'
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