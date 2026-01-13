/**
 * Example: Converting Python SciPy code to JavaScript
 * 
 * This file demonstrates how to convert the Python imports:
 *   from scipy import linalg
 *   from scipy.sparse import lil_matrix, csr_matrix
 *   from scipy.sparse.linalg import eigsh
 * 
 * to JavaScript equivalents.
 */

// JavaScript imports equivalent to Python imports
import * as linalg from './linalg.js';
import { lil_matrix, csr_matrix } from './index.js';
import { eigsh } from './sparse_linalg.js';

console.log('=== scipy-js Example ===\n');

// Example 1: Using lil_matrix
console.log('Example 1: LIL Matrix');
console.log('---------------------');
const A = new lil_matrix([5, 5]);
A.set(0, 0, 1.0);
A.set(1, 1, 2.0);
A.set(2, 2, 3.0);
A.set(0, 4, 0.5);
console.log('Created LIL matrix:', A.toString());
console.log('Number of non-zero elements:', A.nnz);
console.log('Element at (0, 0):', A.get(0, 0));
console.log('Element at (1, 1):', A.get(1, 1));
console.log();

// Example 2: Creating CSR matrix from dense array
console.log('Example 2: CSR Matrix');
console.log('---------------------');
const dense = [
  [4, -1, 0],
  [-1, 4, -1],
  [0, -1, 4]
];
const B = csr_matrix.fromDense(dense);
console.log('Created CSR matrix:', B.toString());
console.log('Matrix data:', B.data);
console.log('Column indices:', B.indices);
console.log('Row pointers:', B.indptr);
console.log();

// Example 3: Matrix-vector multiplication
console.log('Example 3: Matrix-Vector Multiplication');
console.log('---------------------------------------');
const vec = [1, 2, 3];
const result = B.matvec(vec);
console.log('Matrix:');
dense.forEach(row => console.log(row));
console.log('Vector:', vec);
console.log('Result:', result);
console.log();

// Example 4: Linear algebra operations
console.log('Example 4: Linear Algebra Operations');
console.log('------------------------------------');
const v1 = [1, 2, 3];
const v2 = [4, 5, 6];
console.log('Vector 1:', v1);
console.log('Vector 2:', v2);
console.log('Dot product:', linalg.dot(v1, v2));
console.log('Norm of v1:', linalg.norm(v1));
console.log();

const M1 = [[1, 2], [3, 4]];
const M2 = [[5, 6], [7, 8]];
console.log('Matrix 1:', M1);
console.log('Matrix 2:', M2);
console.log('Matrix multiplication:', linalg.matmul(M1, M2));
console.log('Transpose of M1:', linalg.transpose(M1));
console.log();

// Example 5: Solving linear systems
console.log('Example 5: Solving Linear Systems');
console.log('---------------------------------');
const A_sys = [[3, 1], [1, 2]];
const b_sys = [9, 8];
console.log('System: Ax = b');
console.log('A =', A_sys);
console.log('b =', b_sys);
const x = linalg.solve(A_sys, b_sys);
console.log('Solution x =', x);
// Verify: Ax should equal b
const verification = linalg.matvec(A_sys, x);
console.log('Verification Ax =', verification);
console.log();

// Example 6: Sparse eigenvalue problem
console.log('Example 6: Sparse Eigenvalue Problem (eigsh)');
console.log('---------------------------------------------');
// Create a symmetric tridiagonal matrix
const symMatrix = csr_matrix.fromDense([
  [2, -1, 0, 0],
  [-1, 2, -1, 0],
  [0, -1, 2, -1],
  [0, 0, -1, 2]
]);
console.log('Symmetric sparse matrix:', symMatrix.toString());

const eigResult = eigsh(symMatrix, 2, { which: 'LA' });
console.log('Finding 2 largest eigenvalues...');
console.log('Eigenvalues:', eigResult.values);
console.log('First eigenvector:', eigResult.vectors[0]);
console.log('Second eigenvector:', eigResult.vectors[1]);
console.log();

// Example 7: Converting between formats
console.log('Example 7: Converting Between Formats');
console.log('-------------------------------------');
const lilMat = new lil_matrix([3, 3]);
lilMat.set(0, 0, 1);
lilMat.set(1, 1, 2);
lilMat.set(2, 2, 3);
console.log('LIL matrix:', lilMat.toString());
console.log('As dense array:', lilMat.todense());

const csrMat = await lilMat.tocsr();
console.log('Converted to CSR:', csrMat.toString());
console.log('CSR data:', csrMat.data);
console.log();

console.log('=== All examples completed successfully! ===');
