/**
 * Example: Sparse Eigenvalue Problems
 */
import { csr_matrix } from '../index.js';
import { eigsh } from '../sparse_linalg.js';

console.log('========================================');
console.log('Example 3: Sparse Eigenvalue Problems');
console.log('========================================\n');

// Example 1: Simple tridiagonal matrix
console.log('--- Tridiagonal Matrix Eigenvalues ---');
const tridiag = [
  [2, -1, 0, 0, 0],
  [-1, 2, -1, 0, 0],
  [0, -1, 2, -1, 0],
  [0, 0, -1, 2, -1],
  [0, 0, 0, -1, 2]
];

const csr1 = csr_matrix.fromDense(tridiag);
console.log('Matrix (5×5 tridiagonal):');
tridiag.forEach(row => console.log(' ', row));
console.log();

const result1 = eigsh(csr1, 2, { which: 'LA' });
console.log('Finding 2 largest eigenvalues (LA)...');
console.log('Eigenvalues:', result1.values);
console.log('Eigenvector 1:', result1.vectors[0]);
console.log('Eigenvector 2:', result1.vectors[1]);
console.log();

// Example 2: Simple 3×3 symmetric matrix
console.log('--- Simple 3×3 Symmetric Matrix ---');
const symm = [
  [4, -1, 0],
  [-1, 4, -1],
  [0, -1, 4]
];

const csr2 = csr_matrix.fromDense(symm);
console.log('Matrix (3×3 symmetric):');
symm.forEach(row => console.log(' ', row));
console.log();

const result2 = eigsh(csr2, 2, { which: 'LA' });
console.log('Finding 2 largest eigenvalues (LA)...');
console.log('Eigenvalues:', result2.values);
console.log('Eigenvector 1:', result2.vectors[0]);
console.log('Eigenvector 2:', result2.vectors[1]);
console.log();

// Example 3: Verify eigenvalue equation A*v = λ*v
console.log('--- Verifying Eigenvalue Equation A×v = λ×v ---');
const eigenvalue = result2.values[0];
const eigenvector = result2.vectors[0];
const Av = csr2.matvec(eigenvector);
const lambdaV = eigenvector.map(x => x * eigenvalue);

console.log('Eigenvalue λ:', eigenvalue);
console.log('Eigenvector v:', eigenvector);
console.log('A×v:', Av);
console.log('λ×v:', lambdaV);
console.log('Difference:', Av.map((val, i) => Math.abs(val - lambdaV[i])));
console.log();

console.log('✅ Eigenvalue examples completed!\n');
