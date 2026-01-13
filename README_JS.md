# scipy-js

JavaScript port of SciPy linear algebra and sparse matrix functionality.

## Overview

This package provides JavaScript implementations of:
- `scipy.linalg`: Linear algebra functions
- `scipy.sparse.lil_matrix`: List of Lists sparse matrix format
- `scipy.sparse.csr_matrix`: Compressed Sparse Row matrix format
- `scipy.sparse.linalg.eigsh`: Eigenvalue solver for symmetric sparse matrices

## Installation

Since this is a standalone implementation, you can copy the files directly into your project or use them as ES6 modules.

```bash
# No external dependencies required
npm install
```

## Usage

### Importing

```javascript
// Import the main module
import scipy from './index.js';

// Or import specific components
import { linalg, sparse } from './index.js';
import { lil_matrix, csr_matrix } from './index.js';
```

### Python to JavaScript Conversion

The original Python code:
```python
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
```

Becomes in JavaScript:
```javascript
import * as linalg from './linalg.js';
import { lil_matrix, csr_matrix } from './index.js';
import { eigsh } from './sparse_linalg.js';
```

Or using the scipy-like namespace:
```javascript
import scipy from './index.js';

// Access like: scipy.linalg, scipy.sparse.lil_matrix, etc.
const { linalg, sparse } = scipy;
const matrix = new sparse.lil_matrix([10, 10]);
```

### Examples

#### Sparse Matrices

**LIL Matrix (List of Lists)**

```javascript
import { lil_matrix } from './lil_matrix.js';

// Create a 100x100 sparse matrix
const mat = new lil_matrix([100, 100]);

// Set individual elements
mat.set(0, 0, 1.5);
mat.set(5, 10, 2.5);
mat.set(99, 99, 3.5);

// Get elements
console.log(mat.get(0, 0));  // 1.5
console.log(mat.get(1, 1));  // 0 (unset elements are 0)

// Create from dense array
const dense = [
  [1, 0, 2],
  [0, 3, 0],
  [4, 0, 5]
];
const mat2 = new lil_matrix(dense);
console.log(mat2.nnz);  // 5 (number of non-zero elements)

// Convert to dense
const result = mat2.todense();
console.log(result);
```

**CSR Matrix (Compressed Sparse Row)**

```javascript
import { CsrMatrix } from './csr_matrix.js';

// Create from dense array
const mat = CsrMatrix.fromDense([
  [1, 0, 2],
  [0, 3, 0],
  [4, 0, 5]
]);

// Matrix-vector multiplication
const vec = [1, 2, 3];
const result = mat.matvec(vec);
console.log(result);  // [7, 6, 19]

// Access elements
console.log(mat.get(0, 2));  // 2

// Convert to dense
const dense = mat.todense();
```

#### Linear Algebra

```javascript
import * as linalg from './linalg.js';

// Dot product
const a = [1, 2, 3];
const b = [4, 5, 6];
console.log(linalg.dot(a, b));  // 32

// Vector norm
const v = [3, 4];
console.log(linalg.norm(v));  // 5

// Matrix multiplication
const A = [[1, 2], [3, 4]];
const B = [[5, 6], [7, 8]];
const C = linalg.matmul(A, B);
console.log(C);  // [[19, 22], [43, 50]]

// Transpose
const AT = linalg.transpose(A);
console.log(AT);  // [[1, 3], [2, 4]]

// Solve linear system Ax = b
const A2 = [[2, 1], [1, 3]];
const b2 = [5, 6];
const x = linalg.solve(A2, b2);
console.log(x);  // Solution vector
```

#### Sparse Eigenvalue Problems

```javascript
import { CsrMatrix } from './csr_matrix.js';
import { eigsh } from './sparse_linalg.js';

// Create a symmetric sparse matrix
const mat = CsrMatrix.fromDense([
  [2, -1, 0],
  [-1, 2, -1],
  [0, -1, 2]
]);

// Find the 2 largest eigenvalues and eigenvectors
const result = eigsh(mat, 2, { which: 'LA' });

console.log('Eigenvalues:', result.values);
console.log('Eigenvectors:', result.vectors);

// Options for eigsh:
// - which: 'LM' (largest magnitude), 'SM' (smallest magnitude),
//          'LA' (largest algebraic), 'SA' (smallest algebraic)
// - v0: starting vector (optional)
// - tol: convergence tolerance (default: 1e-10)
```

## API Reference

### Sparse Matrix Classes

#### `lil_matrix(shape_or_array, options)`

List of Lists sparse matrix format. Efficient for incremental construction.

- **shape_or_array**: Either `[rows, cols]` or a 2D dense array
- **options.dtype**: Data type (default: 'float64')

Methods:
- `get(i, j)`: Get element at position (i, j)
- `set(i, j, value)`: Set element at position (i, j)
- `todense()`: Convert to dense 2D array
- `tocsr()`: Convert to CSR format
- `nnz`: Property returning number of non-zero elements
- `shape`: Property returning matrix dimensions [rows, cols]

#### `csr_matrix` / `CsrMatrix`

Compressed Sparse Row matrix format. Efficient for arithmetic operations.

Static methods:
- `CsrMatrix.fromDense(array)`: Create from dense 2D array
- `CsrMatrix.fromLil(lil)`: Create from LIL matrix

Methods:
- `get(i, j)`: Get element at position (i, j)
- `matvec(vec)`: Matrix-vector multiplication
- `todense()`: Convert to dense 2D array
- `tolil()`: Convert to LIL format
- `transpose()`: Transpose matrix
- `nnz`: Property returning number of non-zero elements
- `shape`: Property returning matrix dimensions [rows, cols]

### Linear Algebra Functions (`linalg`)

- `dot(a, b)`: Dot product of two vectors
- `norm(v)`: Euclidean (L2) norm of a vector
- `normalize(v)`: Normalize vector to unit length
- `matmul(A, B)`: Matrix-matrix multiplication
- `matvec(A, b)`: Matrix-vector multiplication
- `transpose(A)`: Transpose a matrix
- `eye(n)`: Create n×n identity matrix
- `solve(A, b)`: Solve linear system Ax = b
- `det(A)`: Compute determinant

### Sparse Linear Algebra (`sparse.linalg`)

#### `eigsh(A, k, options)`

Find k eigenvalues and eigenvectors of a symmetric sparse matrix using the Lanczos algorithm.

Parameters:
- `A`: Sparse matrix (must have `matvec` method)
- `k`: Number of eigenvalues to compute (default: 6)
- `options.which`: Which eigenvalues ('LM', 'SM', 'LA', 'SA')
- `options.v0`: Starting vector (optional)
- `options.tol`: Convergence tolerance (default: 1e-10)

Returns:
- Object with `values` (array of eigenvalues) and `vectors` (array of eigenvectors)

## Comparison with SciPy Python API

### Python
```python
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import numpy as np

# Create sparse matrix
A = lil_matrix((100, 100))
A[0, 0] = 1.0
A[5, 10] = 2.0

# Convert to CSR
B = A.tocsr()

# Eigenvalues
vals, vecs = eigsh(B, k=6, which='LM')
```

### JavaScript
```javascript
import * as linalg from './linalg.js';
import { lil_matrix, csr_matrix } from './index.js';
import { eigsh } from './sparse_linalg.js';

// Create sparse matrix
const A = new lil_matrix([100, 100]);
A.set(0, 0, 1.0);
A.set(5, 10, 2.0);

// Convert to CSR
const B = await A.tocsr();

// Eigenvalues
const result = eigsh(B, 6, { which: 'LM' });
const vals = result.values;
const vecs = result.vectors;
```

## Testing

Run the test suite:

```bash
# Run basic tests
npm test

# Run Python comparison tests (validates against scipy output)
node --test tests/test_with_python_data.js
```

### Test Data Generation

The repository includes a Python script that generates test data from scipy for validation:

```bash
cd tests
python3 generate_test_data.py
```

This creates `test_data.json` containing expected outputs from Python's scipy library, which is used by `tests/test_with_python_data.js` to validate the JavaScript implementation.

## Examples

The `examples/` directory contains comprehensive examples demonstrating all features:

```bash
# Run sparse matrices example
node examples/sparse_matrices.js

# Run linear algebra operations example
node examples/linalg_operations.js

# Run eigenvalue problems example
node examples/eigenvalues.js
```

See [`examples/README.md`](examples/README.md) for detailed documentation.

## Implementation Notes

- **No external dependencies**: Pure JavaScript implementation
- **ES6 modules**: Uses modern JavaScript module syntax
- **Simplified algorithms**: Some algorithms are simplified versions for educational purposes
- **Performance**: Not optimized for large-scale problems (for that, use WebAssembly or compiled libraries)

## Limitations

- Dense eigenvalue decomposition (`linalg.eig`) is not fully implemented
- Limited to 2D matrices
- No GPU acceleration
- Simplified Lanczos algorithm for `eigsh` (production code should use more robust implementations)

## License

BSD-3-Clause (same as SciPy)

## Contributing

This is a port of SciPy functionality to JavaScript. For the original SciPy library, see: https://github.com/scipy/scipy
