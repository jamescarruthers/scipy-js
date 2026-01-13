/**
 * scipy-js: JavaScript port of SciPy functionality
 * 
 * This module provides JavaScript implementations of:
 * - scipy.linalg: Linear algebra functions
 * - scipy.sparse: Sparse matrix data structures (lil_matrix, csr_matrix)
 * - scipy.sparse.linalg: Sparse linear algebra functions (eigsh)
 */

// Linear algebra module
export * as linalg from './linalg.js';

// Sparse matrix classes
export { LilMatrix, lil_matrix } from './lil_matrix.js';
export { CsrMatrix, csr_matrix } from './csr_matrix.js';

// Sparse linear algebra
import { eigsh } from './sparse_linalg.js';

// Create sparse object with linalg submodule for scipy-like API
export const sparse = {
  linalg: {
    eigsh
  }
};

// Re-export sparse matrix constructors under sparse namespace
sparse.lil_matrix = (await import('./lil_matrix.js')).lil_matrix;
sparse.csr_matrix = (await import('./csr_matrix.js')).csr_matrix;
sparse.LilMatrix = (await import('./lil_matrix.js')).LilMatrix;
sparse.CsrMatrix = (await import('./csr_matrix.js')).CsrMatrix;

// Default export
export default {
  linalg: await import('./linalg.js'),
  sparse
};
