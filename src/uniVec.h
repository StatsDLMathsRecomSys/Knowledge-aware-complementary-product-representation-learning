/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <random>

#include "args.h"
#include "matrix.h"
#include "model.h"
#include "qmatrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"
#include "dataLoader.h"

namespace uni_vec {

class UniVec {
 protected:

  int64_t ntokens;
  int64_t nViewTokens;
  int64_t nSubTokens;
  int64_t nSearchTokens;
  int64_t expectToken;

  std::shared_ptr<Args> args_;

  std::shared_ptr<int2VecOfInt> item2Word_;

  std::shared_ptr<Matrix> userInput_;
  std::shared_ptr<Matrix> userViewInput_;
  std::shared_ptr<Matrix> userWordOutput_;

  std::shared_ptr<Matrix> itemInput_;
  std::shared_ptr<Matrix> wordOutput_;
  std::shared_ptr<Matrix> itemOutput_;
  std::shared_ptr<Matrix> itemViewOutput_;

  std::shared_ptr<DataLoader> dataLoader_;

  std::shared_ptr<QMatrix> qinput_;
  std::shared_ptr<QMatrix> qoutput_;

  std::shared_ptr<Model> model_;
  std::shared_ptr<Model> exModel_;

  std::atomic<int64_t> tokenCount_{};

  std::atomic<real> loss_{};
  std::atomic<real> lossView_{};
  std::atomic<real> lossSub_{};
  std::atomic<real> lossSearch_{};

  std::default_random_engine rng;

  std::chrono::steady_clock::time_point start_;
  void signModel(std::ostream&);
  bool checkModel(std::istream&);
  void startThreads();
  void addInputVector(Vector&, int32_t) const;
  void regWordModel(Model&, int32_t, const std::vector<int32_t>&, real) ;

  void trainOnObs(Model&, Model&, Model&, const std::vector<int32_t>&, real);
  void trainOnSubObs(Model&, Model&, const std::vector<int32_t>&, real);
  void trainOnSearchObs(Model&, const std::vector<int32_t>&, real);

  void trainThread(int32_t);
  std::vector<std::pair<real, std::string>> getNN(
      const Matrix& wordVectors,
      const Vector& queryVec,
      int32_t k,
      const std::set<std::string>& banSet);
  void lazyComputeWordVectors();
  void printInfo(real, real, std::ostream&);

  bool quant_;
  int32_t version;
  std::unique_ptr<Matrix> wordVectors_;

 public:
  UniVec();

  int32_t getWordId(const std::string& word) const;

  int32_t getSubwordId(const std::string& subword) const;

  void getWordVector(Vector& vec, const std::string& word) const;

  void getSubwordVector(Vector& vec, const std::string& subword) const;

  inline void getInputVector(Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;

  std::shared_ptr<const int2VecOfInt> getItem2Word() const;

  std::shared_ptr<const Matrix> getUserInputMatrix() const;

  std::shared_ptr<const Matrix> getItemInputMatrix() const;

  std::shared_ptr<const Matrix> getItemOutputMatrix() const;

  std::shared_ptr<const Matrix> getWordOutputMatrix() const;

  void saveVectors(const std::string& filename) const;

  void saveVectors(const std::string& filename, std::shared_ptr<const Matrix> mat) const;

  void saveModel();

  void saveModel(const std::string& filename);

  void loadModel(std::istream& in);

  void loadModel(const std::string& filename);

  void getSentenceVector(std::istream& in, Vector& vec);

  void quantize(const Args& qargs);

  std::tuple<int64_t, double, double>
  test(std::istream& in, int32_t k, real threshold = 0.0);

  bool predictLine(
      std::istream& in,
      std::vector<std::pair<real, std::string>>& predictions,
      int32_t k,
      real threshold) const;

  std::vector<std::pair<std::string, Vector>> getNgramVectors(
      const std::string& word) const;

  std::vector<std::pair<real, std::string>> getNN(
      const std::string& word,
      int32_t k);

  std::vector<std::pair<real, std::string>> getAnalogies(
      int32_t k,
      const std::string& wordA,
      const std::string& wordB,
      const std::string& wordC);

  void loadData(std::shared_ptr<DataLoader> dataLoader);

  void init(std::shared_ptr<Args>, std::shared_ptr<DataLoader>);

  void initMatrix();

  void train(const Args& args);

  void loadVectors(const std::string& filename);

  int getDimension() const;

  bool isQuant() const;
};
} // namespace uni_vec
