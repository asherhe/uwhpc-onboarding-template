#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid
{
private:
  // grid elements
  std::vector<double> grid_;

  std::size_t rows_;
  std::size_t cols_;

public:
  Grid(std::size_t rows, std::size_t cols);

  // getters for grid dimensions
  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  // value at row i, col j
  double &operator()(std::size_t i, std::size_t j);
  double operator()(std::size_t i, std::size_t j) const;
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid);

////////////////////////////
// implementation time!!! //
////////////////////////////

Grid::Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), grid_(rows * cols, 0.0)
{
}

double &Grid::operator()(std::size_t i, std::size_t j)
{
  return grid_[i * cols_ + j];
}

double Grid::operator()(std::size_t i, std::size_t j) const
{
  return grid_[i * cols_ + j];
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid)
{
  auto rows = old_grid.rows(), cols = old_grid.cols();

  // handle boundary conditions separately to avoid branching in loop
  for (auto i = 0; i < rows; ++i)
  {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);
  }
  for (auto j = 0; j < cols; ++j)
  {
    new_grid(j, 0) = old_grid(j, 0);
    new_grid(j, rows - 1) = old_grid(j, rows - 1);
  }

  for (auto i = 1; i < rows - 1; ++i)
    for (auto j = 1; j < cols - 1; ++j)
    {
      // weighted avg provided in problem statement
      new_grid(i, j) = 0.5 * old_grid(i, j) +
                       0.125 * (old_grid(i - 1, j) + old_grid(i + 1, j) +
                                old_grid(i, j - 1) + old_grid(i, j + 1));
    }
}
