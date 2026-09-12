#pragma once

#include <cstddef>
#include <vector>

using namespace std;

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid
{
private:
  // grid elements, 64-byte aligned allocation
  alignas(64) vector<double> grid_;

  size_t rows_;
  size_t cols_;

public:
  Grid(size_t rows, size_t cols);

  // getters for grid dimensions
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  // expose pointer to grid data vector
  double *data() { return grid_.data(); }
  const double *data() const { return grid_.data(); }

  // value at row i, col j
  double &operator()(size_t i, size_t j);
  double operator()(size_t i, size_t j) const;
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid);

////////////////////////////
// implementation time!!! //
////////////////////////////

Grid::Grid(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), grid_(rows * cols, 0.0)
{
}

double &Grid::operator()(size_t i, size_t j)
{
  return grid_[i * cols_ + j];
}

double Grid::operator()(size_t i, size_t j) const
{
  return grid_[i * cols_ + j];
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid)
{
  auto rows = old_grid.rows(), cols = old_grid.cols();

  const double *__restrict old_ptr = old_grid.data();
  double *__restrict new_ptr = new_grid.data();
  old_ptr = static_cast<const double *>(__builtin_assume_aligned(old_ptr, 64));
  new_ptr = static_cast<double *>(__builtin_assume_aligned(new_ptr, 64));

  // handle boundary conditions separately to avoid branching in loop
#pragma omp simd aligned(old_ptr, new_ptr : 64)
  for (size_t i = 0; i < rows; ++i)
  {
    new_ptr[i * cols + 0] = old_ptr[i * cols + 0];
    new_ptr[i * cols + cols - 1] = old_ptr[i * cols + cols - 1];
  }
#pragma omp simd aligned(old_ptr, new_ptr : 64)
  for (size_t j = 0; j < cols; ++j)
  {
    new_ptr[j] = old_ptr[j];
    new_ptr[(rows - 1) * cols + j] = old_ptr[(rows - 1) * cols + j];
  }

  for (size_t i = 1; i < rows - 1; ++i)
  {
#pragma omp simd aligned(old_ptr, new_ptr : 64)
    for (size_t j = 1; j < cols - 1; ++j)
    {
      size_t idx = i * cols + j;
      // weighted avg provided in problem statement
      new_ptr[idx] = 0.5 * old_ptr[idx] +
                     0.125 * (old_ptr[idx - cols] + old_ptr[idx + cols] +
                              old_ptr[idx - 1] + old_ptr[idx + 1]);
    }
  }
}
