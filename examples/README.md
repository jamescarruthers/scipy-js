# scipy-js Examples

This directory contains example files demonstrating the usage of scipy-js modules.

## Running the Examples

Each example can be run independently using Node.js:

```bash
# Run sparse matrices example
node examples/sparse_matrices.js

# Run linear algebra operations example
node examples/linalg_operations.js

# Run eigenvalue problems example
node examples/eigenvalues.js
```

## Examples Overview

### 1. Sparse Matrices (`sparse_matrices.js`)

Demonstrates:
- Creating LIL (List of Lists) matrices
- Creating CSR (Compressed Sparse Row) matrices
- Converting between formats
- Matrix-vector multiplication with sparse matrices
- Converting from/to dense arrays

### 2. Linear Algebra Operations (`linalg_operations.js`)

Demonstrates:
- Dot product of vectors
- Vector norms and normalization
- Matrix-matrix multiplication
- Matrix transpose
- Identity matrix creation
- Matrix-vector multiplication
- Solving linear systems (Ax = b)
- Determinant calculation

### 3. Eigenvalue Problems (`eigenvalues.js`)

Demonstrates:
- Computing eigenvalues for sparse symmetric matrices
- Using the `eigsh` function with different options
- Verifying the eigenvalue equation (A×v = λ×v)
- Working with different matrix types (tridiagonal, symmetric)

## Comparison with Python

These examples are designed to be similar to how you would use scipy in Python:

**Python:**
```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

mat = csr_matrix([[4, -1], [-1, 4]])
eigenvalues, eigenvectors = eigsh(mat, k=2)
```

**JavaScript:**
```javascript
import { csr_matrix } from './index.js';
import { eigsh } from './sparse_linalg.js';

const mat = csr_matrix.fromDense([[4, -1], [-1, 4]]);
const result = eigsh(mat, 2);
const eigenvalues = result.values;
const eigenvectors = result.vectors;
```
