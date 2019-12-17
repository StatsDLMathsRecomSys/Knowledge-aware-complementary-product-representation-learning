/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <vector>

// #if defined(__clang__) || defined(__GNUC__)
// #define FASTTEXT_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
// #elif defined(_MSC_VER)
// #define FASTTEXT_DEPRECATED(msg) __declspec(deprecated(msg))
// #else
// #define FASTTEXT_DEPRECATED(msg)
// #endif

namespace uni_vec {

namespace utils {

int64_t size(std::ifstream&);

void seek(std::ifstream&, int64_t);

template <typename T>
bool contains(const std::vector<T>& container, const T& value) {
  return std::find(container.begin(), container.end(), value) !=
      container.end();
}

struct ItemInfo {
  ItemInfo(int32_t numUniqueWord_, int32_t numUniqueItem_):
  numUniqueWord(numUniqueWord_),
  numUniqueItem(numUniqueItem_) {}

  int32_t numUniqueWord;
  int32_t numUniqueItem;
};

struct HistInfo {
  HistInfo(int32_t numUniqueUser_, int32_t numObs_, int32_t numUniqueInputItems_, int32_t numUniqueOutputItems_): 
  numUniqueUser(numUniqueUser_),
  numObs(numObs_),
  numUniqueInputItems(numUniqueInputItems_),
  numUniqueOutputItems(numUniqueOutputItems_) {}

  int32_t numUniqueUser;
  int32_t numObs;
  int32_t numUniqueInputItems;
  int32_t numUniqueOutputItems;
};

} // namespace utils

} // namespace uni_vec
