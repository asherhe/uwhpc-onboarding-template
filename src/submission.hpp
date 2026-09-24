#pragma once

#include <algorithm>
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

// simple bounding box to track the range in which stencilling calculations are
// actually necessary. expected to grow by one every time step.
// this performs especially well with the benchmark test, which uses the
// kCenterBlock initial condition.
struct BBox {
  size_t row_min, col_min, row_max, col_max;

  BBox() : row_min(1), col_min(1), row_max(0), col_max(0) {}
  BBox(size_t row_min, size_t col_min, size_t row_max, size_t col_max)
      : row_min(row_min), col_min(col_min), row_max(row_max), col_max(col_max) {}

  // we take advantage of the fact that Grid::fit_bbox's default empty return
  // value has min > max
  bool is_empty() { return row_min > row_max || col_min > col_max; }

  // expand all borders by 1 activate grid cells that may take on new values
  void grow(size_t rows, size_t cols) {
    row_min = row_min > 0 ? row_min - 1 : 0;
    col_min = col_min > 0 ? col_min - 1 : 0;
    row_max = row_max < rows - 1 ? row_max + 1 : rows - 1;
    col_max = col_max < cols - 1 ? col_max + 1 : cols - 1;
  }
};

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
  size_t stride_;

  // grid array
  // 64-byte aligned allocation for that juicy simd
  double *grid_;

  // bounding box of active heat spread region
  BBox bbox_;
  // whether bbox_ has been initialized with values
  bool has_bbox_;

public:
  Grid(size_t rows, size_t cols)
      : rows_(rows), cols_(cols),
        // round up to multiple of 8 via bit mask
        stride_((cols + 7) & ~7),
        grid_(static_cast<double *>(::operator new[](rows * stride_ * sizeof(double), align_val_t{64}))),
        has_bbox_(false) {}

  ~Grid() { ::operator delete[](grid_, align_val_t(64)); }

  // disable copy construction, copy assignment - each grid_ instance should
  // only be accessed by one Grid at any time.
  Grid(const Grid &) = delete;
  Grid &operator=(const Grid &) = delete;

  // move operations (used by std::swap that is called in main.cpp)
  // NOTE: this is asher again from a day later, it turns out the std::swap doesn't even invoke the move operations
  // because it swaps the POINTERS to Grids :skull:. still good practice to implement this i guess
  Grid(Grid &&other) noexcept : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_), grid_(other.grid_) {
    other.grid_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.stride_ = 0;
  }
  Grid &operator=(Grid &&other) noexcept {
    if (this != &other) {
      // free grid memory about to be replaced
      ::operator delete[](grid_, std::align_val_t{64});

      rows_ = other.rows_;
      cols_ = other.cols_;
      stride_ = other.stride_;
      grid_ = other.grid_;

      other.grid_ = nullptr;
      other.rows_ = 0;
      other.cols_ = 0;
      other.stride_ = 0;
    }
    return *this;
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t stride() const { return stride_; }

  double *data() { return grid_; }
  const double *data() const { return grid_; }

  const BBox &bbox() const { return bbox_; }
  void set_bbox(const BBox &bbox) {
    bbox_ = bbox;
    has_bbox_ = true;
  }
  bool has_bbox() const { return has_bbox_; }

  double &operator()(size_t i, size_t j) noexcept { return grid_[i * stride_ + j]; }
  double operator()(size_t i, size_t j) const noexcept { return grid_[i * stride_ + j]; }

  // determine a bounding box that encloses all nonzero values of the grid
  BBox fit_bbox() const {
    BBox bbox{rows_ - 1, cols_ - 1, 0, 0};

    for (size_t i = 0; i < rows_; ++i) {
      size_t col_min = 0;
      while (col_min < cols_ && operator()(i, col_min) == 0.0)
        ++col_min;

      if (col_min == cols_) continue;

      size_t col_max = cols_ - 1;
      while (col_max > col_min && operator()(i, col_max) == 0.0)
        --col_max;

      bbox.row_min = min(bbox.row_min, i);
      bbox.row_max = max(bbox.row_max, i);
      bbox.col_min = min(bbox.col_min, col_min);
      bbox.col_max = max(bbox.col_max, col_max);
    }

    return bbox;
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  const size_t rows = old_grid.rows(), cols = old_grid.cols(), stride = old_grid.stride();

  const double *RESTRICT old_ptr = old_grid.data();
  double *RESTRICT new_ptr = new_grid.data();

  BBox bbox;
  if (old_grid.has_bbox())
    bbox = old_grid.bbox();
  else {
    bbox = old_grid.fit_bbox();
  }
  bbox.grow(rows, cols);
  new_grid.set_bbox(bbox);

  if (bbox.is_empty()) return;

  // initial pass: reset all values to zero because values outside the
  // bounding box are skipped. we can count on this working because we know
  // old_grid and new_grid are swapped every timestep
  // DEBUG: try doing this outside the loop to see if this is the reason why initial test is always faster
  memset(new_ptr, 0, rows * stride * sizeof(*new_ptr));

  // handle boundary conditions separately to avoid branching in loop
  for (size_t i = bbox.row_min; i <= bbox.row_max; ++i) {
    new_ptr[i * stride] = old_ptr[i * stride];
    new_ptr[i * stride + cols - 1] = old_ptr[i * stride + cols - 1];
  }

  const size_t last_row = (rows - 1) * stride, num_cols = bbox.col_max - bbox.col_min + 1;
  memcpy(new_ptr + bbox.col_min, old_ptr + bbox.col_min, num_cols * sizeof(*old_ptr));
  memcpy(new_ptr + last_row + bbox.col_min, old_ptr + last_row + bbox.col_min, num_cols * sizeof(*old_ptr));

#pragma omp parallel for schedule(static)
  for (size_t i = max(bbox.row_min, size_t{1}); i <= min(bbox.row_max, rows - 2); ++i) {
#pragma omp simd
    for (size_t j = max(bbox.col_min, size_t{1}); j <= min(bbox.col_max, cols - 2); ++j) {
      size_t idx = i * stride + j;
      new_ptr[idx] = 0.5 * old_ptr[idx] +
                     0.125 * (old_ptr[idx - stride] + old_ptr[idx + stride] + old_ptr[idx - 1] + old_ptr[idx + 1]);
    }
  }
}
