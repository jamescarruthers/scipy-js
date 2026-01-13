/**
 * Example: Sparse Matrices (LIL and CSR)
 */
import { lil_matrix, csr_matrix } from '../index.js';

console.log('========================================');
console.log('Example 1: Sparse Matrices');
console.log('========================================\n');

// Create a LIL matrix
console.log('--- Creating a LIL (List of Lists) Matrix ---');
const lil = new lil_matrix([10, 10]);
lil.set(0, 0, 5.0);
lil.set(2, 3, 2.5);
lil.set(5, 7, 3.14);
lil.set(9, 9, 1.0);

console.log('LIL Matrix:', lil.toString());
console.log('Shape:', lil.shape);
console.log('Non-zero elements:', lil.nnz);
console.log('Element at (2, 3):', lil.get(2, 3));
console.log('Element at (0, 1) [zero]:', lil.get(0, 1));
console.log();

// Create from dense array
console.log('--- Creating LIL from Dense Array ---');
const dense = [
  [1, 0, 2],
  [0, 3, 0],
  [4, 0, 5]
];
const lil2 = new lil_matrix(dense);
console.log('Original dense:', dense);
console.log('LIL Matrix:', lil2.toString());
console.log('Back to dense:', lil2.todense());
console.log();

// Convert LIL to CSR
console.log('--- Converting LIL to CSR ---');
const csr = await lil2.tocsr();
console.log('CSR Matrix:', csr.toString());
console.log('CSR data array:', csr.data);
console.log('CSR column indices:', csr.indices);
console.log('CSR row pointers:', csr.indptr);
console.log();

// Create CSR directly from dense
console.log('--- Creating CSR Directly from Dense ---');
const dense2 = [
  [4, -1, 0, 0],
  [-1, 4, -1, 0],
  [0, -1, 4, -1],
  [0, 0, -1, 4]
];
const csr2 = csr_matrix.fromDense(dense2);
console.log('Original dense:', dense2);
console.log('CSR Matrix:', csr2.toString());
console.log();

// Matrix-vector multiplication with CSR
console.log('--- Matrix-Vector Multiplication ---');
const vec = [1, 2, 3, 4];
const result = csr2.matvec(vec);
console.log('Vector:', vec);
console.log('Result:', result);
console.log();

console.log('✅ Sparse matrix examples completed!\n');
