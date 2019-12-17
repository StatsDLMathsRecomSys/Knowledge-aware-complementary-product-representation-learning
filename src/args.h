/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace uni_vec {

enum class model_name : int { cbow = 1, sg, sup };
enum class combine_method: int {concat = 1, mean, meanSum};
enum class loss_name : int { hs = 1, ns, softmax, ova };

class Args {

 public:
  Args();
  std::string input;

  std::string itemWordInput;
  std::string userWordInput;
  
  std::string userHistInput;
  std::string userHistInputView;
  std::string userHistInputSub;
  std::string userHistInputSearch;

  std::string output;
  double lr;
  int lrUpdateRate;
  int dim;
  int userDim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  combine_method combine;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;

  bool saveOutput;
  bool skipContext;
  bool skipUserContext;
  bool skipTrxData;
  bool skipViewData;
  bool skipSubData;
  bool skipSearchData;

  bool shuffleViewData;
  bool shuffleTrxData;

  bool regOutput;
  bool useConcat;
  bool quasiAtten;
  bool qout;
  bool retrain;
  bool qnorm;
  size_t cutoff;
  size_t dsub;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printQuantizationHelp();
  void save(std::ostream&);
  void load(std::istream&);
  void dump(std::ostream&) const;

  std::string lossToString(loss_name) const;
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;
  std::string combineToString(combine_method) const;
};
} // namespace uni_vec
