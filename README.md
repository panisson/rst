# RST - Uniform Random Spanning Trees

RST is a Python package dedicated to extracting uniform random spanning trees from graphs. It leverages the powerful capabilities of the popular libraries Numpy and Scipy, with core methods implemented in Cython to offer significant performance enhancements over pure Python implementations.

Author: André Panisson

## Features

- Extracts uniform random spanning trees from a given graph.
- Utilizes the power and efficiency of Numpy and Scipy for mathematical and scientific computations.
- Core methods implemented in Cython for optimal performance.
- Supports graphs of various sizes and complexities.

## Installation

To install RST, first clone this repository and then install via pip:

```bash
pip install -e .
```

## Usage

Here's a basic example of how to use the RST package:

```python
import rst
import numpy as np
from scipy import sparse

# Define the adjacency matrix of your graph
graph = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
# Transform the adjacency matrix in a sparse matrix
csgraph = sparse.csr_matrix(graph)

# Generate a uniform random spanning tree
tree = rst.random_spanning_tree(csgraph, seed=0)

# `tree` is now an edge index representation of the spanning tree
>>> tree
array([[3, 1, 0],
       [2, 0, 2]], dtype=uint64)
```

For more detailed usage and examples, please refer to the Documentation or check out our Examples folder.

## Contributing
We welcome contributions! Please see our Contribution Guidelines for detailed information on how you can contribute to the project.

## License
This project is licensed under the terms of the MIT license. See LICENSE for further details.

## Contact
Feel free to open an issue if you find a bug or have any suggestions to improve the package. Alternatively, you can reach out at panisson@gmail.com.