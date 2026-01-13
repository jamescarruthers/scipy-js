/**
 * Linear Algebra module
 * 
 * JavaScript implementation of basic scipy.linalg functions
 */

/**
 * Compute the matrix-vector product
 * @param {Array<Array<number>>} A - Matrix
 * @param {Array<number>} b - Vector
 * @returns {Array<number>} Result vector
 */
export function matvec(A, b) {
  if (!A || !A[0] || A[0].length !== b.length) {
    throw new Error('Dimension mismatch');
  }
  
  const result = new Array(A.length);
  for (let i = 0; i < A.length; i++) {
    result[i] = 0;
    for (let j = 0; j < A[i].length; j++) {
      result[i] += A[i][j] * b[j];
    }
  }
  return result;
}

/**
 * Compute the matrix-matrix product
 * @param {Array<Array<number>>} A - First matrix
 * @param {Array<Array<number>>} B - Second matrix
 * @returns {Array<Array<number>>} Result matrix
 */
export function matmul(A, B) {
  const m = A.length;
  const n = B[0].length;
  const p = B.length;
  
  if (A[0].length !== p) {
    throw new Error('Dimension mismatch');
  }
  
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
 * Compute the dot product of two vectors
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Dot product
 */
export function dot(a, b) {
  if (a.length !== b.length) {
    throw new Error('Dimension mismatch');
  }
  
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return result;
}

/**
 * Compute the Euclidean (L2) norm of a vector
 * @param {Array<number>} v - Vector
 * @returns {number} Norm
 */
export function norm(v) {
  return Math.sqrt(dot(v, v));
}

/**
 * Normalize a vector to unit length
 * @param {Array<number>} v - Vector
 * @returns {Array<number>} Normalized vector
 */
export function normalize(v) {
  const n = norm(v);
  if (n === 0) return v.slice();
  return v.map(x => x / n);
}

/**
 * Transpose a matrix
 * @param {Array<Array<number>>} A - Matrix
 * @returns {Array<Array<number>>} Transposed matrix
 */
export function transpose(A) {
  const m = A.length;
  const n = A[0].length;
  const AT = Array.from({ length: n }, () => new Array(m));
  
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      AT[j][i] = A[i][j];
    }
  }
  
  return AT;
}

/**
 * Create an identity matrix
 * @param {number} n - Size of the matrix
 * @returns {Array<Array<number>>} Identity matrix
 */
export function eye(n) {
  const I = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    I[i][i] = 1;
  }
  return I;
}

/**
 * Solve a linear system Ax = b using Gaussian elimination
 * @param {Array<Array<number>>} A - Coefficient matrix
 * @param {Array<number>} b - Right-hand side vector
 * @returns {Array<number>} Solution vector
 */
export function solve(A, b) {
  const n = A.length;
  
  // Create augmented matrix [A|b]
  const aug = A.map((row, i) => [...row, b[i]]);
  
  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) {
        maxRow = k;
      }
    }
    
    // Swap rows
    [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];
    
    // Eliminate column
    for (let k = i + 1; k < n; k++) {
      const factor = aug[k][i] / aug[i][i];
      for (let j = i; j <= n; j++) {
        aug[k][j] -= factor * aug[i][j];
      }
    }
  }
  
  // Back substitution
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= aug[i][j] * x[j];
    }
    x[i] /= aug[i][i];
  }
  
  return x;
}

/**
 * Compute eigenvalues and eigenvectors using QR algorithm (for dense matrices)
 * This is a simplified implementation for educational purposes
 * @param {Array<Array<number>>} A - Square matrix
 * @param {number} maxIter - Maximum iterations (default: 100)
 * @returns {Object} {values: eigenvalues, vectors: eigenvectors}
 */
export function eig(A, maxIter = 100) {
  const n = A.length;
  
  // Simple implementation using power iteration for the largest eigenvalue
  // For a full implementation, we would use QR algorithm or other methods
  
  // This is a placeholder - a full implementation would be much more complex
  throw new Error('Dense eigenvalue decomposition not yet fully implemented. Use eigsh for symmetric matrices.');
}

/**
 * Compute determinant using LU decomposition
 * @param {Array<Array<number>>} A - Square matrix
 * @returns {number} Determinant
 */
export function det(A) {
  const n = A.length;
  const U = A.map(row => [...row]); // Copy matrix
  let sign = 1;
  
  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(U[k][i]) > Math.abs(U[maxRow][i])) {
        maxRow = k;
      }
    }
    
    if (maxRow !== i) {
      [U[i], U[maxRow]] = [U[maxRow], U[i]];
      sign *= -1;
    }
    
    // Eliminate column
    for (let k = i + 1; k < n; k++) {
      const factor = U[k][i] / U[i][i];
      for (let j = i; j < n; j++) {
        U[k][j] -= factor * U[i][j];
      }
    }
  }
  
  // Determinant is product of diagonal elements
  let result = sign;
  for (let i = 0; i < n; i++) {
    result *= U[i][i];
  }
  
  return result;
}

export default {
  matvec,
  matmul,
  dot,
  norm,
  normalize,
  transpose,
  eye,
  solve,
  eig,
  det
};
