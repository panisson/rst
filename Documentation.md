# Documentation

RST is a Python package that leverages two fundamental algorithms for generating Uniform Random Spanning Trees (RSTs). These algorithms are:

1. **Random Walk Method:** This approach involves including each edge $(i,j)$ of a random walk passing from node $i$ to node $j$ whenever node $j$ has not been previously visited, until $n-1$ edges are included in the generated tree.

2. **Wilson Algorithm:** A faster and more sophisticated method, Wilson's algorithm generates an RST by iteratively performing random walks from unvisited nodes selected uniformly at random, deleting loops, and adding the resulting path to the tree. (Wilson, 1996)

Our package implements three key functions for tree generation:

## 1. wilson_random_spanning_tree(graph)
Extract a uniform random spanning tree using the Wilson method.

### Usage
```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.wilson_random_spanning_tree(graph, seed=42)
```

## 2. rw_random_spanning_tree(graph)
Extract a uniform random spanning tree using the Random Walk method.

### Usage

```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.rw_random_spanning_tree(graph, seed=42)
```

## 3. mixed_random_spanning_tree(graph)

This function is a hybrid approach combining the Random Walk and Wilson methods. It first runs the Random Walk method until n edges are traversed, and then switches to the Wilson algorithm.

The idea behind this approach is to address the computational bottlenecks of the two base methods: the initial part for the Wilson algorithm and the latter part for the Random Walk method.

### Usage

```python
import rst

# Define your graph
graph = ...

# Generate the tree
tree = rst.mixed_random_spanning_tree(graph, seed=42)
```

Each of these functions takes a graph representation (such as an adjacency matrix) as input and outputs an edge index (list of edges) representation of the spanning tree, with exactly N-1 edges.

**Note 1:** All functions receive as input a Compressed Sparse Row matrix (see [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)).

**Note 2:** The package assumes that the graph is undirected and connected for correct results. 
If the graph is not connected, the function will keep searching for the unreachable nodes in the random walk processes, so resulting in an infinite loop.

## References

Wilson, D.B. (1996). Generating random spanning trees more quickly than the cover time. Proceedings of the twenty-eighth annual ACM symposium on Theory of computing. pp. 296–303.
