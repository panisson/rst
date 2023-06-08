# Documentation

RST is a Python package that leverages two fundamental algorithms for generating Uniform Random Spanning Trees (RSTs). These algorithms are:

1. **Random Walk Method:** This approach involves including each edge $(i,j)$ of a random walk passing from node $i$ to node $j$ whenever node $j$ has not been previously visited, until $n-1$ edges are included in the generated tree.

2. **Wilson Algorithm:** A faster and more sophisticated method, Wilson's algorithm generates an RST by iteratively performing random walks from unvisited nodes selected uniformly at random, deleting loops, and adding the resulting path to the tree. (Wilson, 1996)

Our package implements three key functions for tree generation:

## 1. random_spanning_tree(graph)
Extract a uniform random spanning tree using the Wilson method.

### Usage
```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.random_spanning_tree(graph, seed=42)
```

## 2. rw_random_spanning_tree(graph)
Extract a uniform random spanning tree using the Random Walk method.

### Usage

```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.rw_random_spanning_tree(graph)
```

## 3. rw_random_spanning_tree_v2(graph)

This function is a hybrid approach combining the Random Walk and Wilson methods. It first runs the Random Walk method until n edges are traversed, and then switches to the Wilson algorithm.

The idea behind this approach is to address the computational bottlenecks of the two base methods: the initial part for the Wilson algorithm and the latter part for the Random Walk method.

### Usage

```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.rw_random_spanning_tree_v2(graph, seed=42)
```

Each of these functions takes a graph representation (such as an adjacency matrix) as input and outputs an edge index (list of edges) representation of the spanning tree, with exactly N-1 edges.

**Note:** The package assumes that the graph is undirected and connected for correct results. 
If the graph is not connected, the function will keep searching for the unreachable nodes in the random walk processes, so resulting in an infinite loop.

## References

Wilson, D.B. (1996). Generating random spanning trees more quickly than the cover time. Proceedings of the twenty-eighth annual ACM symposium on Theory of computing. pp. 296–303.
