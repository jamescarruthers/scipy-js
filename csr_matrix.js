/**
 * Compressed Sparse Row (CSR) matrix format
 * 
 * This is an efficient structure for arithmetic operations and matrix-vector products.
 * Data is stored in three arrays:
 * - data: non-zero values
 * - indices: column indices of non-zero values
 * - indptr: index pointers for each row
 */
export class CsrMatrix {
  /**
   * Create a CSR sparse matrix
   * @param {Object} args - CSR components {data, indices, indptr, shape}
   */
  constructor({ data, indices, indptr, shape }) {
    this.data = data;
    this.indices = indices;
    this.indptr = indptr;
    this.shape = shape;
    this._format = 'csr';
    this.dtype = 'float64';
  }

  /**
   * Create CSR matrix from LIL matrix
   */
  static fromLil(lil) {
    const data = [];
    const indices = [];
    const indptr = [0];
    
    for (let i = 0; i < lil.shape[0]; i++) {
      for (let idx = 0; idx < lil.rows[i].length; idx++) {
        indices.push(lil.rows[i][idx]);
        data.push(lil.data[i][idx]);
      }
      indptr.push(data.length);
    }
    
    return new CsrMatrix({
      data,
      indices,
      indptr,
      shape: lil.shape
    });
  }

  /**
   * Create CSR matrix from dense array
   */
  static fromDense(dense) {
    const data = [];
    const indices = [];
    const indptr = [0];
    const shape = [dense.length, dense[0]?.length || 0];
    
    for (let i = 0; i < dense.length; i++) {
      for (let j = 0; j < dense[i].length; j++) {
        if (dense[i][j] !== 0) {
          indices.push(j);
          data.push(dense[i][j]);
        }
      }
      indptr.push(data.length);
    }
    
    return new CsrMatrix({ data, indices, indptr, shape });
  }

  /**
   * Get element at position (i, j)
   */
  get(i, j) {
    if (i < 0 || i >= this.shape[0] || j < 0 || j >= this.shape[1]) {
      throw new Error('Index out of bounds');
    }
    
    const start = this.indptr[i];
    const end = this.indptr[i + 1];
    
    for (let idx = start; idx < end; idx++) {
      if (this.indices[idx] === j) {
        return this.data[idx];
      }
    }
    
    return 0;
  }

  /**
   * Get number of non-zero elements
   */
  get nnz() {
    return this.data.length;
  }

  /**
   * Convert to dense array
   */
  todense() {
    const dense = Array.from({ length: this.shape[0] }, () =>
      Array(this.shape[1]).fill(0)
    );
    
    for (let i = 0; i < this.shape[0]; i++) {
      const start = this.indptr[i];
      const end = this.indptr[i + 1];
      
      for (let idx = start; idx < end; idx++) {
        const j = this.indices[idx];
        dense[i][j] = this.data[idx];
      }
    }
    
    return dense;
  }

  /**
   * Convert to LIL format
   */
  async tolil() {
    const { LilMatrix } = await import('./lil_matrix.js');
    const lil = new LilMatrix(this.shape);
    
    for (let i = 0; i < this.shape[0]; i++) {
      const start = this.indptr[i];
      const end = this.indptr[i + 1];
      
      lil.rows[i] = this.indices.slice(start, end);
      lil.data[i] = this.data.slice(start, end);
    }
    
    return lil;
  }

  /**
   * Matrix-vector multiplication
   */
  matvec(vec) {
    if (vec.length !== this.shape[1]) {
      throw new Error('Dimension mismatch');
    }
    
    const result = new Array(this.shape[0]).fill(0);
    
    for (let i = 0; i < this.shape[0]; i++) {
      const start = this.indptr[i];
      const end = this.indptr[i + 1];
      
      for (let idx = start; idx < end; idx++) {
        const j = this.indices[idx];
        result[i] += this.data[idx] * vec[j];
      }
    }
    
    return result;
  }

  /**
   * Transpose the matrix (returns CSC which has same structure as CSR)
   */
  transpose() {
    const [M, N] = this.shape;
    const newData = new Array(this.nnz);
    const newIndices = new Array(this.nnz);
    const newIndptr = new Array(N + 1).fill(0);
    
    // Count entries per column
    for (let i = 0; i < this.nnz; i++) {
      newIndptr[this.indices[i] + 1]++;
    }
    
    // Cumulative sum to get indptr
    for (let i = 0; i < N; i++) {
      newIndptr[i + 1] += newIndptr[i];
    }
    
    // Fill in the data
    const counter = [...newIndptr];
    for (let i = 0; i < M; i++) {
      for (let idx = this.indptr[i]; idx < this.indptr[i + 1]; idx++) {
        const j = this.indices[idx];
        const dest = counter[j];
        newIndices[dest] = i;
        newData[dest] = this.data[idx];
        counter[j]++;
      }
    }
    
    return new CsrMatrix({
      data: newData,
      indices: newIndices,
      indptr: newIndptr,
      shape: [N, M]
    });
  }

  /**
   * String representation
   */
  toString() {
    return `CsrMatrix(shape=${this.shape}, nnz=${this.nnz})`;
  }
}

/**
 * Alias for backward compatibility with scipy naming
 */
export const csr_matrix = CsrMatrix;
