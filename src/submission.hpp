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
  // gird elements
  std::vector<std::vector<double>> grid_;

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
