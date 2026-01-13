# scipy-js Tests

This directory contains test files and test data for validating the scipy-js implementation.

## Test Files

### `test_with_python_data.js`

Comprehensive test suite that validates JavaScript implementation against test data generated from Python's scipy library.

**Test Categories:**
1. **Sparse Matrices** - Tests LIL and CSR matrix creation, conversion, and operations
2. **Linear Algebra Operations** - Tests dot product, norm, matmul, transpose, solve, etc.
3. **Sparse Matrix-Vector Multiplication** - Tests CSR matvec operations
4. **Eigenvalue Problems** - Tests sparse eigenvalue solver (eigsh)

**Running:**
```bash
node --test tests/test_with_python_data.js
```

### `generate_test_data.py`

Python script that generates test data using scipy. This creates `test_data.json` which contains expected outputs from Python's scipy library for comparison.

**Running:**
```bash
cd tests
python3 generate_test_data.py
```

**Note:** This script must be run from within the `tests` directory to avoid conflicts with the local `scipy` source directory.

## Test Data

### `test_data.json`

JSON file containing test cases and expected results generated from Python's scipy library. Structure:

```json
{
  "sparse_matrices": [...],      // Tests for LIL and CSR matrices
  "linalg_operations": [...],    // Tests for linear algebra functions
  "sparse_matvec": [...],        // Tests for sparse matrix-vector mult
  "eigsh": [...]                 // Tests for eigenvalue solver
}
```

## Regenerating Test Data

If you modify the JavaScript implementation or want to add new test cases:

1. Edit `generate_test_data.py` to add new test cases
2. Run the script from the tests directory:
   ```bash
   cd tests
   python3 generate_test_data.py
   ```
3. Run the tests to validate:
   ```bash
   cd ..
   node --test tests/test_with_python_data.js
   ```

## Test Tolerances

The tests use different tolerance levels depending on the operation:
- **Basic operations** (dot, norm, etc.): `1e-10`
- **Solve operations**: `1e-9`
- **Eigenvalues**: `1e-4` (due to algorithm differences)
- **Eigenvalue equation verification**: `1e-3`

These tolerances account for:
- Floating-point precision differences
- Algorithm implementation variations
- Iterative solver convergence differences

## Requirements

To regenerate test data, you need:
- Python 3.x
- NumPy: `pip install numpy`
- SciPy: `pip install scipy`

The test script must be run from within the `tests` directory to avoid import conflicts with the scipy source code in the repository root.
