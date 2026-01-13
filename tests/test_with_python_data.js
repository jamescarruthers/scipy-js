/**
 * Tests for scipy-js using Python-generated test data
 * 
 * This test file validates the JavaScript implementation against
 * test data generated from Python's scipy library.
 */

import { test } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// Import scipy-js modules
import { lil_matrix, LilMatrix } from '../lil_matrix.js';
import { csr_matrix, CsrMatrix } from '../csr_matrix.js';
import * as linalg from '../linalg.js';
import { eigsh } from '../sparse_linalg.js';

// Load test data
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const testDataPath = join(__dirname, 'test_data.json');
const testData = JSON.parse(readFileSync(testDataPath, 'utf8'));

// Helper function to compare arrays with tolerance
function assertArraysClose(actual, expected, tolerance = 1e-10, message = '') {
  assert.strictEqual(actual.length, expected.length, 
    `${message} - Array lengths differ: ${actual.length} vs ${expected.length}`);
  
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    assert.ok(diff < tolerance, 
      `${message} - Element ${i}: ${actual[i]} vs ${expected[i]} (diff: ${diff})`);
  }
}

// Helper function to compare 2D arrays with tolerance
function assert2DArraysClose(actual, expected, tolerance = 1e-10, message = '') {
  assert.strictEqual(actual.length, expected.length,
    `${message} - Row count differs: ${actual.length} vs ${expected.length}`);
  
  for (let i = 0; i < actual.length; i++) {
    assert.strictEqual(actual[i].length, expected[i].length,
      `${message} - Column count differs at row ${i}: ${actual[i].length} vs ${expected[i].length}`);
    
    for (let j = 0; j < actual[i].length; j++) {
      const diff = Math.abs(actual[i][j] - expected[i][j]);
      assert.ok(diff < tolerance,
        `${message} - Element [${i}][${j}]: ${actual[i][j]} vs ${expected[i][j]} (diff: ${diff})`);
    }
  }
}

// Sparse Matrix Tests
test('Sparse Matrices - Python comparison', async (t) => {
  for (const testCase of testData.sparse_matrices) {
    await t.test(testCase.name, () => {
      // Test LIL matrix
      const lil = new lil_matrix(testCase.dense);
      assert.deepStrictEqual(lil.shape, testCase.lil.shape, 'LIL shape mismatch');
      assert.strictEqual(lil.nnz, testCase.lil.nnz, 'LIL nnz mismatch');
      assert2DArraysClose(lil.todense(), testCase.lil.todense, 1e-10, 'LIL todense');
      
      // Test CSR matrix
      const csr = CsrMatrix.fromDense(testCase.dense);
      assert.deepStrictEqual(csr.shape, testCase.csr.shape, 'CSR shape mismatch');
      assert.strictEqual(csr.nnz, testCase.csr.nnz, 'CSR nnz mismatch');
      assertArraysClose(csr.data, testCase.csr.data, 1e-10, 'CSR data');
      assert.deepStrictEqual(csr.indices, testCase.csr.indices, 'CSR indices mismatch');
      assert.deepStrictEqual(csr.indptr, testCase.csr.indptr, 'CSR indptr mismatch');
      assert2DArraysClose(csr.todense(), testCase.csr.todense, 1e-10, 'CSR todense');
    });
  }
});

// Linear Algebra Tests
test('Linear Algebra Operations - Python comparison', async (t) => {
  for (const testCase of testData.linalg_operations) {
    await t.test(testCase.name, () => {
      let result;
      
      switch (testCase.operation) {
        case 'dot':
          result = linalg.dot(testCase.inputs.a, testCase.inputs.b);
          assert.ok(Math.abs(result - testCase.expected) < 1e-10,
            `Dot product mismatch: ${result} vs ${testCase.expected}`);
          break;
          
        case 'norm':
          result = linalg.norm(testCase.inputs.v);
          assert.ok(Math.abs(result - testCase.expected) < 1e-10,
            `Norm mismatch: ${result} vs ${testCase.expected}`);
          break;
          
        case 'matmul':
          result = linalg.matmul(testCase.inputs.A, testCase.inputs.B);
          assert2DArraysClose(result, testCase.expected, 1e-10, 'Matmul result');
          break;
          
        case 'transpose':
          result = linalg.transpose(testCase.inputs.A);
          assert2DArraysClose(result, testCase.expected, 1e-10, 'Transpose result');
          break;
          
        case 'solve':
          result = linalg.solve(testCase.inputs.A, testCase.inputs.b);
          assertArraysClose(result, testCase.expected, 1e-9, 'Solve result');
          break;
          
        case 'matvec':
          result = linalg.matvec(testCase.inputs.A, testCase.inputs.b);
          assertArraysClose(result, testCase.expected, 1e-10, 'Matvec result');
          break;
          
        default:
          throw new Error(`Unknown operation: ${testCase.operation}`);
      }
    });
  }
});

// Sparse Matrix-Vector Multiplication Tests
test('Sparse Matrix-Vector Multiplication - Python comparison', async (t) => {
  for (const testCase of testData.sparse_matvec) {
    await t.test(testCase.name, () => {
      const csr = CsrMatrix.fromDense(testCase.matrix.dense);
      const result = csr.matvec(testCase.vector);
      assertArraysClose(result, testCase.expected, 1e-10, 'Sparse matvec result');
    });
  }
});

// Eigenvalue Tests
test('Sparse Eigenvalue Problems (eigsh) - Python comparison', async (t) => {
  for (const testCase of testData.eigsh) {
    await t.test(testCase.name, () => {
      const csr = CsrMatrix.fromDense(testCase.matrix.dense);
      const result = eigsh(csr, testCase.k, { which: testCase.which });
      
      // Check that we got the right number of eigenvalues
      assert.strictEqual(result.values.length, testCase.expected.eigenvalues.length,
        'Number of eigenvalues mismatch');
      
      // Check that we got the right number of eigenvectors
      assert.strictEqual(result.vectors.length, testCase.expected.eigenvectors.length,
        'Number of eigenvectors mismatch');
      
      // Sort eigenvalues for comparison (in descending order for 'LA')
      const sortedActual = [...result.values].sort((a, b) => b - a);
      const sortedExpected = [...testCase.expected.eigenvalues].sort((a, b) => b - a);
      
      // Compare eigenvalues (allow larger tolerance due to algorithm differences)
      assertArraysClose(sortedActual, sortedExpected, 1e-4,
        'Eigenvalues comparison');
      
      // Note: Eigenvectors can differ by a sign, so we just check they're unit vectors
      // and verify the eigenvalue equation A*v = λ*v
      for (let i = 0; i < result.vectors.length; i++) {
        const vec = result.vectors[i];
        const eigenvalue = result.values[i];
        
        // Check that eigenvector is unit length
        const vecNorm = linalg.norm(vec);
        assert.ok(Math.abs(vecNorm - 1.0) < 1e-6,
          `Eigenvector ${i} is not unit length: ${vecNorm}`);
        
        // Check eigenvalue equation: A*v = λ*v
        const Av = csr.matvec(vec);
        const lambdaV = vec.map(x => x * eigenvalue);
        assertArraysClose(Av, lambdaV, 1e-3,
          `Eigenvalue equation A*v = λ*v for eigenvector ${i}`);
      }
    });
  }
});
