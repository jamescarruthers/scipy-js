/**
 * Tests for scipy-js
 */

import { test } from 'node:test';
import assert from 'node:assert';
import { lil_matrix, LilMatrix } from './lil_matrix.js';
import { csr_matrix, CsrMatrix } from './csr_matrix.js';
import * as linalg from './linalg.js';
import { eigsh } from './sparse_linalg.js';

test('LilMatrix - create from shape', () => {
  const mat = new lil_matrix([3, 3]);
  assert.deepStrictEqual(mat.shape, [3, 3]);
  assert.strictEqual(mat.nnz, 0);
});

test('LilMatrix - set and get elements', () => {
  const mat = new lil_matrix([3, 3]);
  mat.set(0, 0, 1.5);
  mat.set(1, 2, 2.5);
  mat.set(2, 1, 3.5);
  
  assert.strictEqual(mat.get(0, 0), 1.5);
  assert.strictEqual(mat.get(1, 2), 2.5);
  assert.strictEqual(mat.get(2, 1), 3.5);
  assert.strictEqual(mat.get(0, 1), 0);
  assert.strictEqual(mat.nnz, 3);
});

test('LilMatrix - create from dense array', () => {
  const dense = [
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
  ];
  const mat = new lil_matrix(dense);
  
  assert.deepStrictEqual(mat.shape, [3, 3]);
  assert.strictEqual(mat.nnz, 5);
  assert.strictEqual(mat.get(0, 0), 1);
  assert.strictEqual(mat.get(0, 2), 2);
  assert.strictEqual(mat.get(1, 1), 3);
});

test('LilMatrix - todense', () => {
  const dense = [
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
  ];
  const mat = new lil_matrix(dense);
  const result = mat.todense();
  
  assert.deepStrictEqual(result, dense);
});

test('CsrMatrix - create from dense', () => {
  const dense = [
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
  ];
  const mat = CsrMatrix.fromDense(dense);
  
  assert.deepStrictEqual(mat.shape, [3, 3]);
  assert.strictEqual(mat.nnz, 5);
  assert.deepStrictEqual(mat.data, [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(mat.indices, [0, 2, 1, 0, 2]);
  assert.deepStrictEqual(mat.indptr, [0, 2, 3, 5]);
});

test('CsrMatrix - get elements', () => {
  const dense = [
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
  ];
  const mat = CsrMatrix.fromDense(dense);
  
  assert.strictEqual(mat.get(0, 0), 1);
  assert.strictEqual(mat.get(0, 2), 2);
  assert.strictEqual(mat.get(1, 1), 3);
  assert.strictEqual(mat.get(2, 0), 4);
  assert.strictEqual(mat.get(0, 1), 0);
});

test('CsrMatrix - todense', () => {
  const dense = [
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
  ];
  const mat = CsrMatrix.fromDense(dense);
  const result = mat.todense();
  
  assert.deepStrictEqual(result, dense);
});

test('CsrMatrix - matvec', () => {
  const mat = CsrMatrix.fromDense([
    [1, 2],
    [3, 4]
  ]);
  const vec = [1, 2];
  const result = mat.matvec(vec);
  
  assert.deepStrictEqual(result, [5, 11]);
});

test('linalg - dot product', () => {
  const a = [1, 2, 3];
  const b = [4, 5, 6];
  const result = linalg.dot(a, b);
  
  assert.strictEqual(result, 32);
});

test('linalg - norm', () => {
  const v = [3, 4];
  const result = linalg.norm(v);
  
  assert.strictEqual(result, 5);
});

test('linalg - matmul', () => {
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  const result = linalg.matmul(A, B);
  
  assert.deepStrictEqual(result, [[19, 22], [43, 50]]);
});

test('linalg - transpose', () => {
  const A = [[1, 2, 3], [4, 5, 6]];
  const result = linalg.transpose(A);
  
  assert.deepStrictEqual(result, [[1, 4], [2, 5], [3, 6]]);
});

test('linalg - eye', () => {
  const I = linalg.eye(3);
  
  assert.deepStrictEqual(I, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]);
});

test('linalg - solve simple system', () => {
  const A = [[2, 1], [1, 3]];
  const b = [5, 6];
  const x = linalg.solve(A, b);
  
  // Verify Ax = b
  const result = linalg.matvec(A, x);
  for (let i = 0; i < b.length; i++) {
    assert.ok(Math.abs(result[i] - b[i]) < 1e-10);
  }
});

test('eigsh - simple symmetric matrix', () => {
  // Create a simple 3x3 symmetric sparse matrix
  const mat = CsrMatrix.fromDense([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
  ]);
  
  const result = eigsh(mat, 2, { which: 'LA' });
  
  assert.strictEqual(result.values.length, 2);
  assert.strictEqual(result.vectors.length, 2);
  
  // Eigenvalues should be real for symmetric matrix
  result.values.forEach(val => {
    assert.ok(typeof val === 'number');
  });
  
  // Each eigenvector should have correct dimension
  result.vectors.forEach(vec => {
    assert.strictEqual(vec.length, 3);
  });
});
