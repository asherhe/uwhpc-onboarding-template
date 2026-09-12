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
