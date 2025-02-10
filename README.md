# CI2024_project-work - Genetic Programming Algorithm for Symbolic Regression

> ⚠️`DISCLAMER`⚠️: this project and so all the code and applied strategies were made by a team of 2 people:
>
> - Martina Plumari (s317612)
> - Daniel Bologna

## Table of contents

- [CI2024\_project-work - Genetic Programming Algorithm for Symbolic Regression](#ci2024_project-work---genetic-programming-algorithm-for-symbolic-regression)
	- [Table of contents](#table-of-contents)
	- [Introduction](#introduction)
	- [Data Structures](#data-structures)
		- [Node](#node)
		- [Tree](#tree)
	- [Results](#results)
	- [Conclusions](#conclusions)
	- [Credits](#credits)

## Introduction

## Data Structures

The Genetic Programming algorithm relies on two primary data structures:

* **Node**: Represents a single node in the syntax tree, which can be a function, a variable, or a constant.
* **Tree**: Represents the entire syntax tree, providing functions for manipulation, mutation, and recombination.

### Node

Each ```Node``` object of this class can represent either a mathematical operator, a variable or a constant. 

Each node mantains references to its parent and to its children, allowing for easy tree traversal.

The ```__call__``` method recursively evaluates the node and its children, ensuring correct computation. To prevent errors from invalid mathematical operations, protective checks are applied:

* **Division** (```np.div```): if the denominator is zero, it is replaced with 1e-6 to avoid division by zero.
* **Logarithm** (```np.log```): 
  * If the argument is zero, it is replaced with 1e-6.
  * If the argument is negative, its absolute value is used.
* **Square Root** (```np.sqrt```): if the argument is negative, its absolute value is used.

These safeguards prevent numerical instability during evaluation.

### Tree

The ```Tree``` class represents the syntax tree and acts as a container for the entire expression. The object contains a reference to the root node and keeps track of the depth of the tree and the total number of its nodes. Also the fitness (MSE) is stored in the tree object.

The Tree class includes methods for:

* **Recombination** (Crossover): Combines two parent trees to generate offspring.
* **Mutation**: Modifies the tree structure through various mutation strategies. In particular we implemented the following mutation strategies:
  * **Subtree Mutation**: replaces a subtree with one of its subtrees
  * **Point Mutation**: replaces a function node with a different random function with the same arity.
  * **Hoist Mutation**: replaces the root node with a random subtree.
  * **Collapse Mutation**: replaces a subtree with a leaf node.
  * **Expansion Mutation**: replaces a leaf node with a new random subtree.
  * **Permutation Mutation**: exchanges two random subtrees in the tree.

## Results

## Conclusions

## Credits