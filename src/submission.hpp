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

#pragma omp parallel for schedule(static)
  for (size_t i = 1; i < rows - 1; ++i) {
#pragma omp simd
    for (size_t j = 1; j < cols - 1; ++j) {
      size_t idx = i * padded_cols + j;
      // weighted avg provided in problem statement
      new_ptr[idx] =
          0.5 * old_ptr[idx] +
          0.125 * (old_ptr[idx - padded_cols] + old_ptr[idx + padded_cols] +
                   old_ptr[idx - 1] + old_ptr[idx + 1]);
    }
  }
}
