#include "submission.hpp"

Grid::Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), grid_(rows, std::vector(cols, 0.0))
{
}

double &Grid::operator()(std::size_t i, std::size_t j)
{
  return grid_[i][j];
}

double Grid::operator()(std::size_t i, std::size_t j) const
{
  return grid_[i][j];
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid)
{
  auto rows = old_grid.rows(), cols = old_grid.cols();

  for (auto i = 0; i < rows; ++i)
    for (auto j = 0; j < cols; ++j)
    {
      auto val = old_grid(i, j);
      if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1)
      {
        new_grid(i, j) = val;
        continue;
      }

      // weighted avg provided in problem statement
      new_grid(i, j) = 0.5 * old_grid(i, j) +
                       0.125 * (old_grid(i - 1, j) + old_grid(i + 1, j) +
                                old_grid(i, j - 1) + old_grid(i, j + 1));
    }
}
