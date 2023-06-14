#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import cython
import numpy as np

cimport numpy as np
from cpython cimport array as arr
from libc.stdlib cimport srand, rand, RAND_MAX

ctypedef np.uint8_t BTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint64_t UITYPE_t

# Returns an integer in the range [0, n).
# Uses rand(), and so is affected-by/affects the same seed.

@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline UITYPE_t randint(UITYPE_t n) nogil noexcept:
    # Chop off all of the values that would cause skew...
    cdef int end = RAND_MAX / n # truncate skew
    end *= n
    
    # ... and ignore results from rand() that fall above that limit.
    # (Worst case the loop condition should succeed 50% of the time,
    # so we can expect to bail out of this loop pretty quickly.)
    cdef int r = rand()
    while r >= end:
        r = rand()
    return r % n

@cython.nonecheck(False)
cdef int _rw_rst(
        np.ndarray[int, ndim=1, mode='c'] indptr,
        np.ndarray[int, ndim=1, mode='c'] indices,
        np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag,
        np.ndarray[UITYPE_t, ndim=2, mode='c'] rst,
        int N,
        int max_cycles=-1) noexcept:
    
    cdef int num_visited, num_rst_edges, current_vtx, next_vtx, next_idx
    cdef int cycles, num_neighs, indptr_start, indptr_end
    
    num_visited = 0
    num_rst_edges = 0
    
    current_vtx = randint(N)
    unvisited_flag[current_vtx] = 0
    num_visited += 1
    cycles = 0
    while num_visited < N:
        cycles += 1
        if max_cycles > 0 and cycles >= max_cycles:
            break
        
        indptr_start = indptr[current_vtx]
        indptr_end = indptr[current_vtx+1]
        
        num_neighs = indptr_end - indptr_start
        next_idx = randint(num_neighs)
        next_vtx = indices[indptr_start + next_idx]
        
        if unvisited_flag[next_vtx]:
            unvisited_flag[next_vtx] = 0
            num_visited += 1
            
            rst[0, num_rst_edges] = current_vtx
            rst[1, num_rst_edges] = next_vtx
            num_rst_edges += 1
            
        current_vtx = next_vtx
    return num_rst_edges

@cython.nonecheck(False)
cdef void _rst_v2(
        np.ndarray[int, ndim=1, mode='c'] indptr,
        np.ndarray[int, ndim=1, mode='c'] indices,
        np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag,
        np.ndarray[UITYPE_t, ndim=2, mode='c'] rst,
        int N, int num_visited, int num_rst_edges) noexcept:
    
    cdef UITYPE_t start_vtx, current_vtx, first_vtx, next_vtx
    cdef UITYPE_t s, t, i, visited_idx
    cdef int num_neighs, indptr_start, indptr_end, next_idx, j
    
    cdef arr.array path  = arr.array('I')
    cdef np.ndarray[ITYPE_t, ndim=1, mode='c'] visit_order
    visit_order = np.arange(N)
    np.random.shuffle(visit_order)
    visited_idx = 0
    
    while num_visited < N:
        
        start_vtx = visit_order[visited_idx]
        visited_idx += 1
        while not unvisited_flag[start_vtx]:
            start_vtx = visit_order[visited_idx]
            visited_idx += 1
        
        current_vtx = start_vtx
        
        arr.resize_smart(path, 1)
        path.data.as_uints[0] = current_vtx
        
        while unvisited_flag[current_vtx]:
            
            indptr_start = indptr[current_vtx]
            indptr_end = indptr[current_vtx+1]

            num_neighs = indptr_end - indptr_start
            next_idx = randint(num_neighs)
            next_vtx = indices[indptr_start + next_idx]
            
            for j in range(len(path)):
                if path.data.as_uints[j] == next_vtx:
                    arr.resize_smart(path, j)
                    break
            
            i = len(path)
            arr.resize_smart(path, i+1)
            path.data.as_uints[i] = next_vtx
            
            current_vtx = next_vtx
            
        s = path.data.as_uints[0]
        for j in range(1, len(path)):
            t = path.data.as_uints[j]
            unvisited_flag[s] = 0
            num_visited += 1
            
            rst[0, num_rst_edges] = s
            rst[1, num_rst_edges] = t
            num_rst_edges += 1
            
            s = t
    
    assert num_rst_edges == N-1, (num_rst_edges, N-1)
    
def wilson_random_spanning_tree(csgraph, int seed=-1):
    """
    Returns a uniform random spanning tree of graph
    using the Wilson algorithm.
    """
    
    cdef UITYPE_t num_visited, N, num_rst_edges
    
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    
    N = csgraph.shape[0]
    rst = np.zeros((2, N-1), dtype=np.uint64)
    
    if seed >= 0:
        np.random.seed(seed)
        srand(seed)
    
    unvisited_flag = np.ones(N, dtype=np.uint8)
    
    
    cdef UITYPE_t first_vtx = randint(N)
    num_visited = 1
    unvisited_flag[first_vtx] = 0
    num_rst_edges = 0
            
    _rst_v2(
        indptr,
        indices,
        unvisited_flag,
        rst,
        N, num_visited, num_rst_edges
        )
    
    return rst
    
def rw_random_spanning_tree(csgraph, int seed=-1):
    """
    Returns a uniform random spanning tree of graph
    using the random walk method.
    """
    cdef int N = csgraph.shape[0]
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    
    unvisited_flag = np.ones(N, dtype=np.uint8)
    rst = np.zeros((2, N-1), dtype=np.uint64)
    
    if seed >= 0:
        np.random.seed(seed)
        srand(seed)
    
    num_rst_edges = _rw_rst(
        indptr,
        indices,
        unvisited_flag,
        rst, N, -1)
    
    return rst

def mixed_random_spanning_tree(csgraph, int seed=-1):
    """
    Returns a uniform random spanning tree of graph.
    This function is a hybrid approach combining the Random Walk and Wilson methods.
    It first runs the Random Walk method until n edges are traversed, 
    and then switches to the Wilson algorithm.
    """
    
    cdef UITYPE_t i, num_visited, N, num_rst_edges
    
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    
    N = csgraph.shape[0]
    rst = np.zeros((2, N-1), dtype=np.uint64)
    
    if seed >= 0:
        np.random.seed(seed)
        srand(seed)
    
    unvisited_flag = np.ones(N, dtype=np.uint8)
    
    num_rst_edges = _rw_rst(
        indptr,
        indices,
        unvisited_flag,
        rst, N, max_cycles=N//8)
    
    num_visited = num_rst_edges + 1
            
    _rst_v2(
        indptr,
        indices,
        unvisited_flag,
        rst,
        N, num_visited, num_rst_edges
        )
    
    return rst