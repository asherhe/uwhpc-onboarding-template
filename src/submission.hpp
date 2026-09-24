#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

using namespace std;

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

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
        // round up to multiple of 8 via bit mask
        padded_cols_((cols + 7) & ~7),
        grid_(static_cast<double *>(::operator new[](
            rows * padded_cols_ * sizeof(double), align_val_t{64}))) {}

  ~Grid() { ::operator delete[](grid_, align_val_t(64)); }

  // disable copy construction, copy assignment - each grid_ instance should
  // only be accessed by one Grid at any time.
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
      // free grid memory about to be replaced
      ::operator delete[](grid_, std::align_val_t{64});

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

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t padded_cols() const { return padded_cols_; }

  double *data() { return grid_; }
  const double *data() const { return grid_; }

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

  const double *RESTRICT old_ptr = old_grid.data();
  double *RESTRICT new_ptr = new_grid.data();

  // handle boundary conditions separately to avoid branching in loop
  for (size_t i = 0; i < rows; ++i) {
    new_ptr[i * padded_cols] = old_ptr[i * padded_cols];
    new_ptr[i * padded_cols + cols - 1] = old_ptr[i * padded_cols + cols - 1];
  }

  const size_t last_row = (rows - 1) * padded_cols;
  memcpy(new_ptr, old_ptr, cols * sizeof(*old_ptr));
  memcpy(new_ptr + last_row, old_ptr + last_row, cols * sizeof(*old_ptr));

#pragma omp parallel for schedule(static)
  for (size_t i = 1; i < rows - 1; ++i) {
#pragma omp simd
    for (size_t j = 1; j < cols - 1; ++j) {
      size_t idx = i * padded_cols + j;
      new_ptr[idx] =
          0.5 * old_ptr[idx] +
          0.125 * (old_ptr[idx - padded_cols] + old_ptr[idx + padded_cols] +
                   old_ptr[idx - 1] + old_ptr[idx + 1]);
    }
  }
}
