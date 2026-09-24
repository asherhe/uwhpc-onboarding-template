#pragma once

#include <cstddef>
#include <vector>

using namespace std;

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  size_t rows_;
  size_t cols_;
  // column count rounded up to multiple of 8. ensures grid_ column sections are
  // stored as multiples of 8 * sizeof(double) = 64 bytes for simd
  size_t padded_cols_;

  // grid array
  // 64-byte aligned allocation for that juicy simd
  double *grid_;

public:
  Grid(size_t rows, size_t cols)
      : rows_(rows), cols_(cols),
        // round up to multiple of 8 with a bit mask
        padded_cols_((cols + 7) & ~7),
        grid_(static_cast<double *>(::operator new[](
            rows * padded_cols_ * sizeof(double), align_val_t{64}))) {}

  ~Grid() { ::operator delete[](grid_, align_val_t(64)); }

  // disable copy construction, copy assignment
  Grid(const Grid &) = delete;
  Grid &operator=(const Grid &) = delete;

  // move operations (used by std::swap that is called in main.cpp)
  Grid(Grid &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_),
        padded_cols_(other.padded_cols_), grid_(other.grid_) {
    other.grid_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.padded_cols_ = 0;
  }
  Grid &operator=(Grid &&other) noexcept {
    if (this != &other) {
      ::operator delete[](grid_, align_val_t{64}); // free existing memory

      rows_ = other.rows_;
      cols_ = other.cols_;
      padded_cols_ = other.padded_cols_;
      grid_ = other.grid_;

      other.grid_ = nullptr;
      other.rows_ = 0;
      other.cols_ = 0;
      other.padded_cols_ = 0;
    }
    return *this;
  }

  // getters for grid dimensions
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t padded_cols() const { return padded_cols_; }

  // expose pointer to grid data array
  double *data() { return grid_; }
  const double *data() const { return grid_; }

  // value at row i, col j
  double &operator()(size_t i, size_t j) { return grid_[i * padded_cols_ + j]; }
  double operator()(size_t i, size_t j) const {
    return grid_[i * padded_cols_ + j];
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  const size_t rows = old_grid.rows(), cols = old_grid.cols(),
               padded_cols = old_grid.padded_cols();

  const double *__restrict old_ptr = old_grid.data();
  double *__restrict new_ptr = new_grid.data();

  // handle boundary conditions separately to avoid branching in loop
  for (size_t i = 0; i < rows; ++i) {
    new_ptr[i * padded_cols] = old_ptr[i * padded_cols];
    new_ptr[i * padded_cols + cols - 1] = old_ptr[i * padded_cols + cols - 1];
  }

  const size_t last_row = (rows - 1) * padded_cols;
#pragma omp simd aligned(old_ptr, new_ptr : 64)
  for (size_t j = 0; j < cols; ++j) {
    new_ptr[j] = old_ptr[j];
    new_ptr[last_row + j] = old_ptr[last_row + j];
  }

  // teh execution of the main stencil loop is divided into blocks. this ensures
  // that most grid retrieval operations can remain in cache for its neighbors.
  const size_t BLOCK_I = 8, BLOCK_J = 128;

  // i ran these tests on battery power so scores are slightly worse
  // apparently running on battery has a bigger impact on my solution than the
  // reference solution.
  // BI,  BJ | Score
  // --------+------
  // 16,  16 | 1.756
  // 32,  16 | 1.610
  // 64,  16 | 1.407
  // 16,  32 | 1.841
  // 64,  32 | 1.537
  // 16,  64 | 1.881
  //  2,  64 | 1.868
  //  4,  64 | 1.882
  //  8,  64 | 1.858
  // 16,  64 | 1.822
  //  4, 128 | 1.809
  //  8, 128 | 1.891 (best)
  // 16, 128 | 1.813
  // 16, 256 | 1.767

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i_blk = 1; i_blk < rows - 1; i_blk += BLOCK_I)
    for (size_t j_blk = 1; j_blk < cols - 1; j_blk += BLOCK_J)
      // not collapsed because we want each block to run on one thread
      for (size_t i = i_blk; i < min(i_blk + BLOCK_I, rows - 1); ++i) {
#pragma omp simd
        for (size_t j = j_blk; j < min(j_blk + BLOCK_J, cols - 1); ++j) {
          size_t idx = i * padded_cols + j;
          // weighted avg provided in problem statement
          new_ptr[idx] =
              0.5 * old_ptr[idx] +
              0.125 * (old_ptr[idx - padded_cols] + old_ptr[idx + padded_cols] +
                       old_ptr[idx - 1] + old_ptr[idx + 1]);
        }
      }
}
