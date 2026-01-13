/**
 * List of Lists (LIL) sparse matrix format
 * 
 * This is an efficient structure for constructing sparse matrices incrementally.
 * Each row is stored as a sorted list of column indices and corresponding values.
 */
export class LilMatrix {
  /**
   * Create a LIL sparse matrix
   * @param {Array|number} arg1 - Either a shape [rows, cols] or a dense 2D array
   * @param {Object} options - Optional parameters
   * @param {string} options.dtype - Data type (default: 'float64')
   */
  constructor(arg1, options = {}) {
    this.dtype = options.dtype || 'float64';
    this._format = 'lil';
    
    if (Array.isArray(arg1) && arg1.length === 2 && typeof arg1[0] === 'number') {
      // Initialize from shape
      const [rows, cols] = arg1;
      this.shape = [rows, cols];
      this.rows = Array.from({ length: rows }, () => []);
      this.data = Array.from({ length: rows }, () => []);
    } else if (Array.isArray(arg1)) {
      // Initialize from dense array
      this._fromDense(arg1);
    } else {
      throw new Error('Unsupported lil_matrix constructor usage');
    }
  }

  /**
   * Initialize from a dense 2D array
   * @private
   */
  _fromDense(dense) {
    this.shape = [dense.length, dense[0]?.length || 0];
    this.rows = [];
    this.data = [];
    
    for (let i = 0; i < dense.length; i++) {
      const rowIndices = [];
      const rowData = [];
      
      for (let j = 0; j < dense[i].length; j++) {
        if (dense[i][j] !== 0) {
          rowIndices.push(j);
          rowData.push(dense[i][j]);
        }
      }
      
      this.rows.push(rowIndices);
      this.data.push(rowData);
    }
  }

  /**
   * Get element at position (i, j)
   */
  get(i, j) {
    if (i < 0 || i >= this.shape[0] || j < 0 || j >= this.shape[1]) {
      throw new Error('Index out of bounds');
    }
    
    const rowIndices = this.rows[i];
    const idx = rowIndices.indexOf(j);
    
    return idx >= 0 ? this.data[i][idx] : 0;
  }

  /**
   * Set element at position (i, j)
   */
  set(i, j, value) {
    if (i < 0 || i >= this.shape[0] || j < 0 || j >= this.shape[1]) {
      throw new Error('Index out of bounds');
    }
    
    const rowIndices = this.rows[i];
    const idx = rowIndices.indexOf(j);
    
    if (value === 0) {
      // Remove the element if it exists
      if (idx >= 0) {
        rowIndices.splice(idx, 1);
        this.data[i].splice(idx, 1);
      }
    } else {
      if (idx >= 0) {
        // Update existing element
        this.data[i][idx] = value;
      } else {
        // Insert new element in sorted order
        const insertIdx = rowIndices.findIndex(col => col > j);
        if (insertIdx === -1) {
          rowIndices.push(j);
          this.data[i].push(value);
        } else {
          rowIndices.splice(insertIdx, 0, j);
          this.data[i].splice(insertIdx, 0, value);
        }
      }
    }
  }

  /**
   * Get number of non-zero elements
   */
  get nnz() {
    return this.data.reduce((sum, row) => sum + row.length, 0);
  }

  /**
   * Convert to CSR (Compressed Sparse Row) format
   */
  async tocsr() {
    const { CsrMatrix } = await import('./csr_matrix.js');
    return CsrMatrix.fromLil(this);
  }

  /**
   * Convert to dense array
   */
  todense() {
    const dense = Array.from({ length: this.shape[0] }, () =>
      Array(this.shape[1]).fill(0)
    );
    
    for (let i = 0; i < this.shape[0]; i++) {
      for (let idx = 0; idx < this.rows[i].length; idx++) {
        const j = this.rows[i][idx];
        dense[i][j] = this.data[i][idx];
      }
    }
    
    return dense;
  }

  /**
   * String representation
   */
  toString() {
    return `LilMatrix(shape=${this.shape}, nnz=${this.nnz})`;
  }
}

/**
 * Alias for backward compatibility with scipy naming
 */
export const lil_matrix = LilMatrix;
