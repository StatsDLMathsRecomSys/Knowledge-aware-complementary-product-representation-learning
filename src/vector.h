/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "real.h"

namespace uni_vec {

class Matrix;
class QMatrix;

class Vector {
 protected:
  std::vector<real> data_;

 public:
  explicit Vector(int64_t);
  Vector(const Vector&) = delete;
  Vector(Vector&&) noexcept;
  Vector& operator=(const Vector&) = delete;
  Vector& operator=(Vector&&);
  friend std::ostream &operator<<( std::ostream &, const Vector &);

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }
  inline real& operator[](int64_t i) {
    return data_[i];
  }
  inline const real& operator[](int64_t i) const {
    return data_[i];
  }

  inline int64_t size() const {
    return data_.size();
  }
  void zero();
  void mul(real);
  real norm() const;
  void addVector(const Vector& source);
  void addVector(const Vector&, real);
  void addRow(const Matrix&, int64_t);
  void addRow(const QMatrix&, int64_t);
  void addRow(const Matrix&, int64_t, real);
  void mul(const QMatrix&, const Vector&);
  void mul(const Matrix&, const Vector&);
  int64_t argmax();
};

std::ostream& operator<<(std::ostream&, const Vector&);

} // namespace uni_vec
