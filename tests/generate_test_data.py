#!/usr/bin/env python3
"""
Generate test data using Python's scipy library.
This data will be used to validate the JavaScript implementation.
"""

import json
import numpy as np
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


def convert_to_serializable(obj):
    """Convert numpy arrays and types to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def generate_sparse_matrix_tests():
    """Generate test cases for sparse matrices."""
    tests = []
    
    # Test 1: Simple 3x3 sparse matrix
    dense1 = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)
    lil1 = lil_matrix(dense1)
    csr1 = lil1.tocsr()
    
    tests.append({
        'name': 'simple_3x3_sparse',
        'dense': dense1.tolist(),
        'lil': {
            'shape': list(lil1.shape),
            'nnz': int(lil1.nnz),
            'todense': lil1.todense().tolist()
        },
        'csr': {
            'shape': list(csr1.shape),
            'nnz': int(csr1.nnz),
            'data': csr1.data.tolist(),
            'indices': csr1.indices.tolist(),
            'indptr': csr1.indptr.tolist(),
            'todense': csr1.todense().tolist()
        }
    })
    
    # Test 2: Larger sparse matrix (5x5) with diagonal
    dense2 = np.diag([1.5, 2.5, 3.5, 4.5, 5.5])
    lil2 = lil_matrix(dense2)
    csr2 = lil2.tocsr()
    
    tests.append({
        'name': 'diagonal_5x5',
        'dense': dense2.tolist(),
        'lil': {
            'shape': list(lil2.shape),
            'nnz': int(lil2.nnz),
            'todense': lil2.todense().tolist()
        },
        'csr': {
            'shape': list(csr2.shape),
            'nnz': int(csr2.nnz),
            'data': csr2.data.tolist(),
            'indices': csr2.indices.tolist(),
            'indptr': csr2.indptr.tolist(),
            'todense': csr2.todense().tolist()
        }
    })
    
    # Test 3: Tridiagonal matrix
    n = 4
    dense3 = np.zeros((n, n))
    for i in range(n):
        dense3[i, i] = 2
        if i > 0:
            dense3[i, i-1] = -1
        if i < n-1:
            dense3[i, i+1] = -1
    
    lil3 = lil_matrix(dense3)
    csr3 = lil3.tocsr()
    
    tests.append({
        'name': 'tridiagonal_4x4',
        'dense': dense3.tolist(),
        'lil': {
            'shape': list(lil3.shape),
            'nnz': int(lil3.nnz),
            'todense': lil3.todense().tolist()
        },
        'csr': {
            'shape': list(csr3.shape),
            'nnz': int(csr3.nnz),
            'data': csr3.data.tolist(),
            'indices': csr3.indices.tolist(),
            'indptr': csr3.indptr.tolist(),
            'todense': csr3.todense().tolist()
        }
    })
    
    return tests


def generate_linalg_tests():
    """Generate test cases for linear algebra operations."""
    tests = []
    
    # Test 1: Dot product
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([5, 6, 7, 8])
    tests.append({
        'name': 'dot_product',
        'operation': 'dot',
        'inputs': {'a': v1.tolist(), 'b': v2.tolist()},
        'expected': float(np.dot(v1, v2))
    })
    
    # Test 2: Vector norm
    v3 = np.array([3, 4])
    tests.append({
        'name': 'norm_3_4_vector',
        'operation': 'norm',
        'inputs': {'v': v3.tolist()},
        'expected': float(np.linalg.norm(v3))
    })
    
    v4 = np.array([1, 2, 3])
    tests.append({
        'name': 'norm_1_2_3_vector',
        'operation': 'norm',
        'inputs': {'v': v4.tolist()},
        'expected': float(np.linalg.norm(v4))
    })
    
    # Test 3: Matrix multiplication
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    tests.append({
        'name': 'matmul_2x3_3x2',
        'operation': 'matmul',
        'inputs': {'A': A.tolist(), 'B': B.tolist()},
        'expected': np.matmul(A, B).tolist()
    })
    
    # Test 4: Transpose
    M = np.array([[1, 2, 3], [4, 5, 6]])
    tests.append({
        'name': 'transpose_2x3',
        'operation': 'transpose',
        'inputs': {'A': M.tolist()},
        'expected': M.T.tolist()
    })
    
    # Test 5: Solve linear system
    A_sys = np.array([[3, 1], [1, 2]], dtype=float)
    b_sys = np.array([9, 8], dtype=float)
    x_sol = linalg.solve(A_sys, b_sys)
    tests.append({
        'name': 'solve_2x2_system',
        'operation': 'solve',
        'inputs': {'A': A_sys.tolist(), 'b': b_sys.tolist()},
        'expected': x_sol.tolist()
    })
    
    # Test 6: Matrix-vector multiplication
    A_mv = np.array([[1, 2], [3, 4], [5, 6]])
    v_mv = np.array([7, 8])
    tests.append({
        'name': 'matvec_3x2_2',
        'operation': 'matvec',
        'inputs': {'A': A_mv.tolist(), 'b': v_mv.tolist()},
        'expected': np.dot(A_mv, v_mv).tolist()
    })
    
    return tests


def generate_sparse_matvec_tests():
    """Generate test cases for sparse matrix-vector multiplication."""
    tests = []
    
    # Test 1: Simple sparse matvec
    dense = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
    csr = csr_matrix(dense)
    vec = np.array([1, 2, 3], dtype=float)
    result = csr.dot(vec)
    
    tests.append({
        'name': 'sparse_matvec_3x3',
        'matrix': {
            'dense': dense.tolist(),
            'data': csr.data.tolist(),
            'indices': csr.indices.tolist(),
            'indptr': csr.indptr.tolist()
        },
        'vector': vec.tolist(),
        'expected': result.tolist()
    })
    
    # Test 2: Larger sparse matvec
    n = 5
    dense2 = np.diag([2]*n) + np.diag([-1]*(n-1), 1) + np.diag([-1]*(n-1), -1)
    csr2 = csr_matrix(dense2)
    vec2 = np.array([1, 0, 1, 0, 1], dtype=float)
    result2 = csr2.dot(vec2)
    
    tests.append({
        'name': 'sparse_matvec_5x5_tridiagonal',
        'matrix': {
            'dense': dense2.tolist(),
            'data': csr2.data.tolist(),
            'indices': csr2.indices.tolist(),
            'indptr': csr2.indptr.tolist()
        },
        'vector': vec2.tolist(),
        'expected': result2.tolist()
    })
    
    return tests


def generate_eigsh_tests():
    """Generate test cases for sparse eigenvalue problems."""
    tests = []
    
    # Test 1: Simple symmetric tridiagonal matrix
    n = 5
    dense = np.zeros((n, n))
    for i in range(n):
        dense[i, i] = 2
        if i > 0:
            dense[i, i-1] = -1
        if i < n-1:
            dense[i, i+1] = -1
    
    csr = csr_matrix(dense)
    
    # Compute 2 largest eigenvalues
    k = 2
    eigenvalues, eigenvectors = eigsh(csr, k=k, which='LA')
    
    tests.append({
        'name': 'eigsh_tridiagonal_5x5_LA',
        'matrix': {
            'dense': dense.tolist(),
            'data': csr.data.tolist(),
            'indices': csr.indices.tolist(),
            'indptr': csr.indptr.tolist()
        },
        'k': k,
        'which': 'LA',
        'expected': {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.T.tolist()  # Transpose to get vectors as rows
        }
    })
    
    # Test 2: Smaller 3x3 symmetric matrix  
    dense2 = np.array([
        [4, -1, 0],
        [-1, 4, -1],
        [0, -1, 4]
    ], dtype=float)
    
    csr2 = csr_matrix(dense2)
    eigenvalues2, eigenvectors2 = eigsh(csr2, k=2, which='LA')
    
    tests.append({
        'name': 'eigsh_simple_3x3_LA',
        'matrix': {
            'dense': dense2.tolist(),
            'data': csr2.data.tolist(),
            'indices': csr2.indices.tolist(),
            'indptr': csr2.indptr.tolist()
        },
        'k': 2,
        'which': 'LA',
        'expected': {
            'eigenvalues': eigenvalues2.tolist(),
            'eigenvectors': eigenvectors2.T.tolist()
        }
    })
    
    return tests


def main():
    """Generate all test data and save to JSON file."""
    test_data = {
        'sparse_matrices': generate_sparse_matrix_tests(),
        'linalg_operations': generate_linalg_tests(),
        'sparse_matvec': generate_sparse_matvec_tests(),
        'eigsh': generate_eigsh_tests()
    }
    
    # Save to JSON file
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Test data generated successfully!")
    print(f"- Sparse matrix tests: {len(test_data['sparse_matrices'])}")
    print(f"- Linear algebra tests: {len(test_data['linalg_operations'])}")
    print(f"- Sparse matvec tests: {len(test_data['sparse_matvec'])}")
    print(f"- Eigsh tests: {len(test_data['eigsh'])}")
    print("\nTest data saved to test_data.json")


if __name__ == '__main__':
    main()
