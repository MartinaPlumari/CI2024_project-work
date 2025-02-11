# CI2024_project-work - Genetic Programming Algorithm for Symbolic Regression

> ⚠️`DISCLAIMER`⚠️: this project and so all the code and applied strategies were made by a team of 2 people:
>
> - Martina Plumari (s317612)
> - Daniel Bologna (s310582)

## Table of contents

- [CI2024\_project-work - Genetic Programming Algorithm for Symbolic Regression](#ci2024_project-work---genetic-programming-algorithm-for-symbolic-regression)
	- [Table of contents](#table-of-contents)
	- [Introduction](#introduction)
	- [Features](#features)
	- [Data Structures](#data-structures)
		- [Node](#node)
		- [Tree](#tree)
	- [Genetic Algorithm](#genetic-algorithm)
		- [Optimization process](#optimization-process)
		- [Hyper-Modern Approach](#hyper-modern-approach)
		- [Dynamic Mutation Probability](#dynamic-mutation-probability)
		- [Guard System to Prevent Bloating](#guard-system-to-prevent-bloating)
	- [Settings](#settings)
	- [Results](#results)
	- [Conclusions and Future Work](#conclusions-and-future-work)
	- [Credits](#credits)

## Introduction

In this project, we developed a Genetic Programming (GP) algorithm for symbolic regression, using evolutionary techniques to iteratively optimize mathematical expressions. 
The algorithm begins with a population of randomly generated expressions and improves them over generations through **mutation**, **crossover**, and **selection**.


## Features

The model is fully customizable and allows changing multiple settings **directly from the command line** (es. dataset path, population size, etc...).

For more information use:

```
python.exe .\main.py --help
```

To start a training session with the **pre-configured settings** use:

```
python.exe .\main.py --dset "<path-to-dataset>" --name "<name-surname>" --id "<student-id>"
```

This will load and process only the first problem from the given folder. Then, at the end it will automatically generate a `.py` file with the final solution.

> **WARNING**: re-running the code will overwrite the solutions.

## Data Structures

The Genetic Programming algorithm relies on two primary data structures:

* **Node**: Represents a single node in the syntax tree, which can be a function, a variable, or a constant.
* **Tree**: Represents the entire syntax tree, providing functions for manipulation, mutation, and recombination.

### Node

Each ```Node``` object of this class can represent either a mathematical operator, a variable or a constant. 

Each node maintains references to its parent and to its children, allowing for easy tree traversal.

The ```__call__``` method recursively evaluates the node and its children, ensuring correct computation. To prevent errors from invalid mathematical operations, protective checks are implemented:

* **Division** (```np.div```): if the denominator is zero, it is replaced with 1e-6 to avoid division by zero.
* **Logarithm** (```np.log```): 
  * If the argument is zero, it is replaced with 1e-6.
  * If the argument is negative, its absolute value is used.
* **Square Root** (```np.sqrt```): if the argument is negative, its absolute value is used.

These safeguards prevent numerical instability during evaluation.

Even if the node could store any callable function, this implementation is limited to NumPy functions. 
In particular, the following functions are supported: 

* ```np.add```, ```np.subtract```, ```np.multiply```, ```np.divide```, ```np.sqrt```, ```np.log```
* ```np.tan```, ```np.sin```, ```np.cos```

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

## Genetic Algorithm

In this project, we developed a dedicated class, `Symreg`, to manage the entire symbolic regression process. The core functionality is encapsulated within `Symreg.train()`, which executes the evolutionary algorithm.

### Optimization process

We implemented a genetic programming approach for the optimization process. The algorithm begins by generating a population of randomly created formulas, each representing a potential relationship between the known independent variables (`x[i]`) and the corresponding result (`y`). The goal is to evolve these formulas over time to produce increasingly accurate results. In each generation, the fittest individuals are selected based on their performance, and genetic operations, such as mutation or recombination, are applied to create a new population, refining the solutions. 

For the generation of the initial population, we experimented with **full**, **grow** and **half-half** methods, but we found that the **full** method worked best in terms of results. However, we left the option to choose the preferred method to the user.
### Hyper-Modern Approach

We adopted a hyper-modern approach for adjusting the population. At each step, while generating offspring, there is a specified probability to mutate an individual or perform recombination between two parents, but not both at the same time.

We also experimented with multiple mutation variants, as discussed in the lectures, implementing the most common mutation types. Users can either select a specific mutation type or opt for a random selection on each iteration. For recombination, we used a single crossover approach, where two parent trees exchange subtrees to generate new individuals.

### Dynamic Mutation Probability

To further enhance the optimization, we implemented a dynamic adjustment of mutation probability:

- Increase the probability if the algorithm is stuck on a plateau for a reasonable number of iterations.
- Reset the probability every time the solution improves.

```
...
# every 50 iterations
if i % 50 == 0:
	checks if the best solution has remained unchanged for long enough
		then increase mutation probability by 0.05
	if it takes too long to improve 
		then stop early
else:
	reset mutation probability to 0.05
...

```

### Guard System to Prevent Bloating

To prevent bloating in the population, we added a guard system that forces tree collapse during mutation if the tree height exceeds a predefined threshold. This ensures that the individuals remain manageable in size.

```
if individual._h < 2:
	# if the tree is too small try expanding the tree
	individual = t.expansion_mutation(individual)
	return individual
elif individual._h >= 4:
	# if the tree is too deep try reducing its size
	individual = t.hoist_mutation(individual)
	return individual
```

## Settings

These are our configurations of the model related to the following results. You can run the code shown earlier to use these settings.

| Setting                      | Value |
|------------------------------|-------|
| Population Size              | 5000  |
| Offspring Size               | 5000  |
| Max Generations              | 1000  |
| Use Random Mutations         | True  |
| Tournament Selection         | 3     |
| Initial Mutation Probability | 0.05  |
| Population Initialization    | 1     |

## Results

| PROBLEM # | BEST RESULT         |
|-----------|---------------------|
| 0         | -2.29435e-05        |
| 1         | -7.62668e-34        |
| 2         | -29277347609686.277 |
| 3         | -516.23614          |
| 4         | -0.32133            |
| 5         | -5.57281e-18        |
| 6         | -0.242187           |
| 7         | -582.14875          |
| 8         | -8981174.25265      |

As we can see, the algorithm was able to find a good solution for most of the small problems. The algorithm is able to find a good solution within relatively few generations, and the dynamic mutation probability helps to prevent stagnation and improve the results.
An additional note is that the algorithm is able to find a good solution even with a relatively small population size, which can be beneficial in terms of computational resources.

## Conclusions and Future Work

This project successfully implemented a Genetic Programming (GP) algorithm for symbolic regression, demonstrating its effectiveness in evolving mathematical expressions to fit given datasets.

The algorithm generally worked well, optimizing mathematical expressions and delivering solid results on many test cases. However, it’s worth noting that in some tests the fitness trend suggested that a solution might not be achievable, like in the case of Problem 2.
Despite the strong performance on smaller problems, challenges remain with more complex expressions and larger datasets, suggesting that scalability might be a problem.

That being said, we focused only on the hyper-modern strategy and did not explore other alternatives. In the future, it might be worthwhile to explore different techniques: such as alternative crossover and selection methods or improved elitism strategies.

## Credits

* The ```Node``` class was based on the implementation by [Prof. Giovanni Squillero](https://github.com/squillero/computational-intelligence/tree/master/2024-25/symreg/gxgp) and only slightly modified and extended to fit our needs.
* The [gplearn](https://gplearn.readthedocs.io/en/stable/intro.html) documentation was a very useful resource for understanding the genetic programming concepts and the implementation of the algorithm.