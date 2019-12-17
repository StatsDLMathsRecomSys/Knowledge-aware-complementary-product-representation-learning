/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "matrix.h"

#include <exception>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>  

#include "utils.h"
#include "vector.h"

namespace uni_vec {

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(int64_t m, int64_t n) : data_(m * n), m_(m), n_(n) {}

std::ostream &operator<<( std::ostream &output, const Matrix &mat ) {
  output << std::fixed; 
  output << std::setprecision(5);
  for (int i = 0; i < mat.rows(); i++) {
      for (int j = 0; j < mat.cols(); j++) {
        if (mat.at(i, j) >= 0) output << " ";
        output << mat.at(i, j) << " ";
      }
      if (i != mat.rows() - 1) output << std::endl;
    }
  return output;            
}

void Matrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

real Matrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec[j];
  }
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

real Matrix::matSelectDot(const Matrix& a, const Matrix& b, const int64_t aPos, const int64_t bPos) {
  assert(aPos >= 0);
  assert(aPos < a.rows());
  assert(bPos >= 0);
  assert(bPos <= b.rows());
  assert(a.cols() == b.cols());
  real score = 0.0;
  for (int64_t j = 0; j < a.cols(); j++) {
    score += a.at(aPos, j) * b.at(bPos, j);
  }
  if (std::isnan(score)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return score;
}

void Matrix::addRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void Matrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void Matrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real Matrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return std::sqrt(norm);
}

void Matrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

void Matrix::save(std::ostream& out) {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = std::vector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

} // namespace uni_vec