# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import numbers
import numpy as np
import random as rnd
from typing import Callable
from utils.arity import arity
import copy
import json

class Node:
    _function: Callable
    _successors: list['Node']
    _parent: 'Node'
    _arity: int #questa mi serve per capire come strutturare i successori
    _str: str #penso sia una sorta di etichetta (stringa) per il nodo
    _leaf: bool #questo mi serve per capire se il nodo è una foglia
    _type: str #questo mi serve per capire se il nodo è una variabile, una costante o una funzione


    def __init__(self, node=None, successors= None, *, name=None):
        
        self._parent = None
        
        #FIRST CASE: the node is a function (or an operator, the two cases can coincide when using np)
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
            
            # assert self._arity is None or len(self._successors) == self._arity, (
            #     "Panic: Incorrect number of children."
            #     + f" Expected {len(tuple(successors))} found {arity(node)}"
            # )
            
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
    

    def __deepcopy__(self, memo):
        """Crea una copia profonda del nodo e dei suoi successori."""
        if id(self) in memo:
            return memo[id(self)]
        
        # Creiamo un nuovo nodo in base al tipo
        if self._type == 'c':  # Costante
            copied_node = Node(float(self._str))
        elif self._type == 'v':  # Variabile
            copied_node = Node(self._str)
        else:  # Funzione
            copied_node = Node(eval(self.short_name), successors=[], name=self._str)  # Mantiene il nome
        
        # Copiamo gli attributi fondamentali
        copied_node._arity = self._arity
        copied_node._type = self._type
        copied_node._leaf = self._leaf
        copied_node._str = self._str
        copied_node._parent = None  # La parent verrà aggiornata in seguito
        
        # Copia ricorsiva dei successori
        copied_node._successors = [copy.deepcopy(s, memo) for s in self._successors]
        
        # Aggiorniamo i riferimenti al parent nei successori copiati
        for child in copied_node._successors:
            child._parent = copied_node
        
        # Memorizziamo la copia nel memo per evitare duplicazioni
        memo[id(self)] = copied_node
        return copied_node
    
    def copy(self):
        # Crea una copia del nodo corrente senza parent e successori

        if self._function is None:
            return None
        
        new_node = Node(
            node=self._function, 
            successors=[], 
            name=self._str
        )
        new_node._type = self._type
        new_node._arity = self._arity
        new_node._leaf = self._leaf
        
        # Copia i successori ricorsivamente
        new_node._successors = [child.copy() for child in self._successors]
        
        # Imposta il parent per i nuovi successori
        for child in new_node._successors:
            child._parent = new_node
        
        return new_node