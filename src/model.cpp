/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "utils.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace uni_vec {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wi_1,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Matrix> wo_1,
    std::shared_ptr<Args> args,
    bool wordModel,
    int32_t seed)

    : hidden_(args->dim),
      exHidden_(args->dim + args->userDim),
      output_(wo->size(0)),
      exOutput_(wo->size(0)),
      grad_(args->dim),
      gradUser_(args->dim),
      exGrad_(args->dim + args->userDim),
      rng(seed),
      quant_(false){
  // I_i
  wi_ = wi;
  ii_ = wi;
  // W_o
  wo_ = wo;
  // U_i
  wi_1_ = wi_1;
  ui_ = wi_1;
  // I_o
  wo_1_ = wo_1;
  io_ = wo_1;

  args_ = args;

  if (wordModel) {
    osz_ = wo->size(0);
  }  else {
    osz_ = io_->size(0);
  }

  hsz_ = args->dim;
  
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  t_log_.reserve(LOG_TABLE_SIZE + 1);
  initSigmoid();
  initLog();

}

void Model::setQuantizePointer(
    std::shared_ptr<QMatrix> qwi,
    std::shared_ptr<QMatrix> qwo,
    bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  wo_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::binaryLogisticConcat(int32_t target, bool label, real lr) {
  real score = sigmoid(io_->dotRow(exHidden_, target));
  real alpha = lr * (real(label) - score);
  exGrad_.addRow(*io_, target, alpha);

  io_->addRow(exHidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::binaryLogisticMean(int32_t target, bool label, real lr) {
  real score = sigmoid(io_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*io_, target, alpha);

  io_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::binaryLogisticMeanSum(int32_t targetIdx, int32_t userIdx, bool label, real lr,
Vector& itemGrad, Vector& userGrad) {
  real userItemScore = Matrix::matSelectDot(*ii_, *ui_, targetIdx, userIdx);
  real itemInOutScore = io_->dotRow(hidden_, targetIdx);

  real score = sigmoid(userItemScore + itemInOutScore);
  real alpha = lr * (real(label) - score);

  itemGrad.addRow(*io_, targetIdx, alpha);
  userGrad.addRow(*ii_, targetIdx, alpha);

  // only update I_o by hidden, skip update I_i
  io_->addRow(hidden_, targetIdx, alpha);

  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}

real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutput(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
}

void Model::computeOutputSigmoid(Vector& hidden, Vector& output) const {
  computeOutput(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = sigmoid(output[i]);
  }
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  computeOutput(hidden, output);
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}

real Model::oneVsAll(const std::vector<int32_t>& targets, real lr) {
  real loss = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    bool isMatch = utils::contains(targets, i);
    loss += binaryLogistic(i, isMatch, lr);
  }

  return loss;
}

void Model::computeConcat(const std::vector<int32_t>& user_hist, int32_t user_pos, 
int32_t item_pos, Vector& exHidden_) {
  assert(exHidden_.size() == ui_->cols() + ii_->cols());
  assert(user_pos == int32_t(1));
  assert(item_pos == int32_t(0));

  exHidden_.zero();
  const int32_t user_idx = user_hist[user_pos];
  const int32_t ui_ncols = ui_->cols();
  const int32_t ii_ncols = ii_->cols();

  // add U_i
  if (!args_->skipUserContext) {
    for (int32_t idx = 0; idx < ui_ncols; ++idx) {
      exHidden_[idx] += ui_->at(user_idx, idx);
    }
  }

 // add mean I_i for all listed items
  for (int32_t pos = 0; pos < user_hist.size(); ++pos) {
    if (pos == user_pos || pos == item_pos) continue;
    int32_t item_hist_idx = user_hist[pos];
    for (int32_t idx = 0; idx < ii_ncols; ++idx) {
      exHidden_[ui_ncols + idx] += ii_->at(item_hist_idx, idx);
    }
  }
  real inv_hist_item_size = 1.0 / (real)(user_hist.size() - 2);

  for (int32_t idx = 0; idx < ii_ncols; ++idx) {
    exHidden_[ui_ncols + idx] *= inv_hist_item_size;
  }
}

void Model::computeMean(const std::vector<int32_t>& user_hist, int32_t user_pos, 
int32_t item_pos, Vector& hidden, bool inputItemOnly) {
  assert(hidden.size() == ui_->cols());
  assert(hidden.size() == ii_->cols());
  assert(user_pos == int32_t(1));
  assert(item_pos == int32_t(0));

  hidden.zero();
  const int32_t user_idx = user_hist[user_pos];
  if (!inputItemOnly) {
    // add U_i
    for (int32_t idx = 0; idx < hidden.size(); ++idx) {
      hidden[idx] += ui_->at(user_idx, idx);
    }
  }
 // add I_i for all listed items
  for (int32_t pos = 0; pos < user_hist.size(); ++pos) {
    if (pos == user_pos || pos == item_pos) continue;
    int32_t item_hist_idx = user_hist[pos];
    for (int32_t idx = 0; idx < hidden.size(); ++idx) {
      hidden[idx] += ii_->at(item_hist_idx, idx);
    }
  }
  real inv_hist_size = 1.0 / (real)(user_hist.size() - 1  - (int)inputItemOnly);
  for (int32_t idx = 0; idx < hidden.size(); ++idx) {
    hidden[idx] *= inv_hist_size;
  }
}

void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden)
    const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(
    const std::pair<real, int32_t>& l,
    const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

void Model::findKBest(
    int32_t k,
    real threshold,
    std::vector<std::pair<real, int32_t>>& heap,
    Vector& hidden,
    Vector& output) const {
  if (args_->loss == loss_name::ova) {
    computeOutputSigmoid(hidden, output);
  } else {
    computeOutputSoftmax(hidden, output);
  }
  for (int32_t i = 0; i < osz_; i++) {
    if (output[i] < threshold) {
      continue;
    }
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

real Model::computeLoss(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr) {
  real loss = 0.0;

  if (args_->loss == loss_name::ns) {
    loss = negativeSampling(targets[targetIndex], lr);
  } else if (args_->loss == loss_name::hs) {
    loss = hierarchicalSoftmax(targets[targetIndex], lr);
  } else if (args_->loss == loss_name::softmax) {
    loss = softmax(targets[targetIndex], lr);
  } else if (args_->loss == loss_name::ova) {
    loss = oneVsAll(targets, lr);
  } else {
    throw std::invalid_argument("Unhandled loss function for this model.");
  }

  return loss;
}


real Model::computeConcatLoss(const std::vector<int32_t>& user_hist, int32_t user_pos, 
int32_t item_pos, real lr) {
  int32_t item_output_idx = user_hist[item_pos];
  real loss = 0.0;
  exGrad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogisticConcat(item_output_idx, true, lr);
    } else {
      loss += binaryLogisticConcat(getNegative(item_output_idx), false, lr);
    }
  }
  return loss;
}


real Model::computeMeanLoss(const std::vector<int32_t>& user_hist, int32_t user_pos, 
int32_t item_pos, real lr) {
  int32_t item_output_idx = user_hist[item_pos];
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogisticMean(item_output_idx, true, lr);
    } else {
      loss += binaryLogisticMean(getNegative(item_output_idx), false, lr);
    }
  }
  return loss;
}

void Model::updateConcat(
    const std::vector<int32_t>& user_hist,
    int32_t user_pos,
    int32_t item_pos,
    real lr
    ) {
  assert(user_hist.size() > 2);
  computeConcat(user_hist, user_pos, item_pos, exHidden_);
  loss_ += computeConcatLoss(user_hist, user_pos, item_pos, lr);

  nexamples_ += 1;

  const int32_t user_idx = user_hist[user_pos];
  const int32_t ui_ncols = ui_->cols();
  const int32_t ii_ncols = ii_->cols();
  const real inv_hist_item_size = 1.0 / (real)(user_hist.size() - 2);

  if (!args_->skipUserContext) {
    for (int32_t col_idx = 0; col_idx < ui_ncols; col_idx++) {
      ui_->at(user_idx, col_idx) += exGrad_[col_idx];
    }
  }

  for (int32_t col_idx = 0; col_idx < ii_ncols; col_idx++) {
    exGrad_[ui_ncols + col_idx] *= inv_hist_item_size;
  }

  for (int32_t pos = 0; pos < user_hist.size(); pos++) {
    if (pos == user_pos || pos == item_pos) continue;
    int32_t item_input_index = user_hist[pos];
    for (int32_t col_idx = 0; col_idx < ii_ncols; col_idx++) {
      ii_->at(item_input_index, col_idx) += exGrad_[ui_ncols + col_idx];
    }
  }
}

void Model::updateMean(
    const std::vector<int32_t>& user_hist,
    int32_t user_pos,
    int32_t item_pos,
    real lr
    ) {
  assert(user_hist.size() > 2);
  computeMean(user_hist, user_pos, item_pos, hidden_, false);
  loss_ += computeMeanLoss(user_hist, user_pos, item_pos, lr);

  nexamples_ += 1;

  // devide by the 1 + num_items = hist.size() - 1
  const int32_t user_idx = user_hist[user_pos];
  const real inv_hist_item_size = 1.0 / (real)(user_hist.size() - 1);
  for (int32_t col_idx = 0; col_idx < grad_.size(); col_idx++) {
    grad_[col_idx] *= inv_hist_item_size;
  }

  ui_->addRow(grad_, user_idx, 1.0);

 // add gard to item input
  for (int32_t pos = 0; pos < user_hist.size(); pos++) {
    if (pos == user_pos || pos == item_pos) continue;

    ii_->addRow(grad_, user_hist[pos], 1.0);
  }
}

void Model::updateMeanSum(
    const std::vector<int32_t>& user_hist,
    int32_t user_pos,
    int32_t item_pos,
    real lr
    ) {
  assert(user_hist.size() > 2);
  computeMean(user_hist, user_pos, item_pos, hidden_, true);

  const int32_t item_output_idx = user_hist[item_pos];
  const int32_t userIdx = user_hist[user_pos];

  real loss = 0.0;
  grad_.zero();
  gradUser_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogisticMeanSum(item_output_idx, userIdx, true, lr, grad_, gradUser_);
    } else {
      loss += binaryLogisticMeanSum(getNegative(item_output_idx), userIdx, false, lr, grad_, gradUser_);
    }
  }

  loss_ += loss;
  nexamples_ += 1;

  // devide by the num_items = hist.size() - 2
  const real inv_hist_item_size = 1.0 / (real)(user_hist.size() - 2);
  for (int32_t col_idx = 0; col_idx < grad_.size(); col_idx++) {
    grad_[col_idx] *= inv_hist_item_size;
  }

  for (int32_t pos = 0; pos < user_hist.size(); pos++) {
    if (pos == user_pos || pos == item_pos) continue;
    ii_->addRow(grad_, user_hist[pos], 1.0);
  }

  ui_->addRow(gradUser_, userIdx, 1.0);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, hidden_);

  assert (targetIndex != kAllLabelsAsTarget);
  assert(targetIndex >= 0);
  assert(targetIndex < osz_);
  loss_ += computeLoss(targets, targetIndex, lr);

  nexamples_ += 1;

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  std::cout << counts.size() << ", " << osz_ << std::endl;
  assert(counts.size() <= osz_);
  assert (args_->loss == loss_name::ns);
  initTableNegatives(counts);
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives_.push_back(i);
    }
  }
  std::shuffle(negatives_.begin(), negatives_.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives_[negpos];
    negpos = (negpos + 1) % negatives_.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }
}

void Model::initLog() {
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

} // namespace uni_vec
