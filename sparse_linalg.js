/**
 * Sparse Linear Algebra module
 * 
 * JavaScript implementation of scipy.sparse.linalg functions
 */

import { dot, norm, normalize } from './linalg.js';

/**
 * Find k eigenvalues and eigenvectors of a symmetric sparse matrix
 * 
 * This implements a simplified version of the Lanczos algorithm for 
 * finding eigenvalues of symmetric matrices.
 * 
 * @param {CsrMatrix|Function} A - Sparse matrix or linear operator
 * @param {number} k - Number of eigenvalues to compute (default: 6)
 * @param {Object} options - Additional options
 * @param {string} options.which - Which eigenvalues to find ('LM' = largest magnitude, 'SM' = smallest magnitude, 'LA' = largest algebraic, 'SA' = smallest algebraic)
 * @param {Array<number>} options.v0 - Starting vector
 * @param {number} options.maxiter - Maximum number of iterations
 * @param {number} options.tol - Convergence tolerance
 * @returns {Object} {values: eigenvalues array, vectors: eigenvectors matrix}
 */
export function eigsh(A, k = 6, options = {}) {
  const {
    which = 'LM',
    v0 = null,
    maxiter = null,
    tol = 1e-10
  } = options;
  
  // Determine matrix size
  let n;
  let matvec;
  
  if (typeof A === 'function') {
    // A is a linear operator
    throw new Error('Linear operator interface not yet implemented');
  } else if (A.matvec) {
    // A is a sparse matrix
    n = A.shape[0];
    matvec = (v) => A.matvec(v);
  } else {
    throw new Error('A must be a sparse matrix with matvec method or a function');
  }
  
  if (k >= n) {
    throw new Error('k must be less than matrix size');
  }
  
  const m = Math.min(2 * k + 1, n); // Lanczos dimension
  const iterations = maxiter || Math.max(2 * m, 100);
  
  // Initialize starting vector
  let v = v0 ? [...v0] : randomVector(n);
  v = normalize(v);
  
  // Lanczos algorithm to build tridiagonal matrix
  const alpha = [];
  const beta = [];
  const V = [v]; // Lanczos vectors
  
  let w = matvec(v);
  alpha.push(dot(w, v));
  
  for (let j = 0; j < m - 1; j++) {
    // w = A * v_j - alpha_j * v_j - beta_{j-1} * v_{j-1}
    w = w.map((val, i) => val - alpha[j] * V[j][i]);
    
    if (j > 0) {
      w = w.map((val, i) => val - beta[j - 1] * V[j - 1][i]);
    }
    
    const betaJ = norm(w);
    beta.push(betaJ);
    
    if (betaJ < tol) {
      // Lucky breakdown - we've found an invariant subspace
      break;
    }
    
    const vNew = w.map(x => x / betaJ);
    V.push(vNew);
    
    w = matvec(vNew);
    alpha.push(dot(w, vNew));
  }
  
  // Solve the tridiagonal eigenvalue problem
  const T = buildTridiagonal(alpha, beta);
  const { values: eigenvalues, vectors: eigenvectors } = solveTridiagonalEig(T);
  
  // Select k eigenvalues based on 'which' parameter
  const indices = selectEigenvalues(eigenvalues, k, which);
  
  // Compute the corresponding eigenvectors in the original space
  const selectedValues = indices.map(i => eigenvalues[i]);
  const selectedVectors = indices.map(i => {
    const y = eigenvectors[i];
    const vec = new Array(n).fill(0);
    
    for (let j = 0; j < V.length; j++) {
      for (let idx = 0; idx < n; idx++) {
        vec[idx] += y[j] * V[j][idx];
      }
    }
    
    return vec;
  });
  
  return {
    values: selectedValues,
    vectors: selectedVectors
  };
}

/**
 * Generate a random vector
 * @private
 */
function randomVector(n) {
  return Array.from({ length: n }, () => Math.random() - 0.5);
}

/**
 * Build tridiagonal matrix from alpha and beta arrays
 * @private
 */
function buildTridiagonal(alpha, beta) {
  const n = alpha.length;
  const T = Array.from({ length: n }, () => new Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    T[i][i] = alpha[i];
    if (i < n - 1) {
      T[i][i + 1] = beta[i];
      T[i + 1][i] = beta[i];
    }
  }
  
  return T;
}

/**
 * Solve eigenvalue problem for tridiagonal matrix using QR algorithm
 * @private
 */
function solveTridiagonalEig(T) {
  const n = T.length;
  let A = T.map(row => [...row]); // Copy matrix
  let Q_total = Array.from({ length: n }, (_, i) => 
    Array.from({ length: n }, (_, j) => i === j ? 1 : 0)
  );
  
  const maxIter = 100;
  
  for (let iter = 0; iter < maxIter; iter++) {
    // QR decomposition
    const { Q, R } = qrDecomposition(A);
    
    // A = R * Q
    A = matmul(R, Q);
    
    // Accumulate Q matrices for eigenvectors
    Q_total = matmul(Q_total, Q);
    
    // Check for convergence (off-diagonal elements small)
    let converged = true;
    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(A[i][i + 1]) > 1e-10) {
        converged = false;
        break;
      }
    }
    
    if (converged) break;
  }
  
  // Extract eigenvalues from diagonal
  const values = A.map((row, i) => row[i]);
  
  // Eigenvectors are columns of Q_total
  const vectors = [];
  for (let i = 0; i < n; i++) {
    vectors.push(Q_total.map(row => row[i]));
  }
  
  return { values, vectors };
}

/**
 * QR decomposition using Gram-Schmidt
 * @private
 */
function qrDecomposition(A) {
  const m = A.length;
  const n = A[0].length;
  const Q = [];
  const R = Array.from({ length: n }, () => new Array(n).fill(0));
  
  for (let j = 0; j < n; j++) {
    // Extract column j
    let v = A.map(row => row[j]);
    
    // Orthogonalize against previous columns
    for (let i = 0; i < j; i++) {
      R[i][j] = dot(Q[i], v);
      v = v.map((val, idx) => val - R[i][j] * Q[i][idx]);
    }
    
    R[j][j] = norm(v);
    Q.push(v.map(x => x / R[j][j]));
  }
  
  // Convert Q to matrix form
  const Q_matrix = Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) => Q[j][i])
  );
  
  return { Q: Q_matrix, R };
}

/**
 * Matrix multiplication helper
 * @private
 */
function matmul(A, B) {
  const m = A.length;
  const n = B[0].length;
  const p = B.length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < p; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  
  return C;
}

/**
 * Select k eigenvalues based on which parameter
 * @private
 */
function selectEigenvalues(eigenvalues, k, which) {
  const indices = eigenvalues.map((val, idx) => ({ val, idx }));
  
  switch (which) {
    case 'LM': // Largest magnitude
      indices.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
      break;
    case 'SM': // Smallest magnitude
      indices.sort((a, b) => Math.abs(a.val) - Math.abs(b.val));
      break;
    case 'LA': // Largest algebraic
      indices.sort((a, b) => b.val - a.val);
      break;
    case 'SA': // Smallest algebraic
      indices.sort((a, b) => a.val - b.val);
      break;
    default:
      throw new Error(`Unknown which parameter: ${which}`);
  }
  
  return indices.slice(0, k).map(x => x.idx);
}

export default {
  eigsh
};
