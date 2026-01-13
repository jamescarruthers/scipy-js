/**
 * Example: Linear Algebra Operations
 */
import * as linalg from '../linalg.js';

console.log('========================================');
console.log('Example 2: Linear Algebra Operations');
console.log('========================================\n');

// Dot product
console.log('--- Dot Product ---');
const v1 = [1, 2, 3, 4];
const v2 = [5, 6, 7, 8];
console.log('v1:', v1);
console.log('v2:', v2);
console.log('v1 · v2 =', linalg.dot(v1, v2));
console.log();

// Vector norm
console.log('--- Vector Norm ---');
const v3 = [3, 4];
console.log('v:', v3);
console.log('||v|| =', linalg.norm(v3));

const v4 = [1, 2, 3];
console.log('v:', v4);
console.log('||v|| =', linalg.norm(v4));
console.log();

// Normalize vector
console.log('--- Normalize Vector ---');
const v5 = [3, 4];
const v5_normalized = linalg.normalize(v5);
console.log('Original:', v5);
console.log('Normalized:', v5_normalized);
console.log('Norm of normalized:', linalg.norm(v5_normalized));
console.log();

// Matrix multiplication
console.log('--- Matrix Multiplication ---');
const A = [[1, 2, 3], [4, 5, 6]];
const B = [[7, 8], [9, 10], [11, 12]];
console.log('A =', A);
console.log('B =', B);
console.log('A × B =', linalg.matmul(A, B));
console.log();

// Transpose
console.log('--- Transpose ---');
const M = [[1, 2, 3], [4, 5, 6]];
console.log('M =', M);
console.log('M^T =', linalg.transpose(M));
console.log();

// Identity matrix
console.log('--- Identity Matrix ---');
const I = linalg.eye(4);
console.log('I(4) =');
I.forEach(row => console.log(' ', row));
console.log();

// Matrix-vector multiplication
console.log('--- Matrix-Vector Multiplication ---');
const A2 = [[1, 2], [3, 4], [5, 6]];
const v6 = [7, 8];
console.log('A =', A2);
console.log('v =', v6);
console.log('A × v =', linalg.matvec(A2, v6));
console.log();

// Solve linear system
console.log('--- Solve Linear System Ax = b ---');
const A_sys = [[3, 1], [1, 2]];
const b_sys = [9, 8];
console.log('A =', A_sys);
console.log('b =', b_sys);
const x = linalg.solve(A_sys, b_sys);
console.log('x =', x);
console.log('Verification A×x =', linalg.matvec(A_sys, x));
console.log();

// Determinant
console.log('--- Determinant ---');
const M2 = [[2, 1], [1, 3]];
console.log('M =', M2);
console.log('det(M) =', linalg.det(M2));
console.log();

console.log('✅ Linear algebra examples completed!\n');
