/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "uniVec.h"

#include "cnpy/cnpy.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace uni_vec {

constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r);

UniVec::UniVec() : quant_(false), wordVectors_(nullptr) {}

std::shared_ptr<const int2VecOfInt> UniVec::getItem2Word() const {
  return item2Word_;
}

const Args UniVec::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const Matrix> UniVec::getUserInputMatrix() const {
  return userInput_;
}

std::shared_ptr<const Matrix> UniVec::getItemInputMatrix() const {
  return itemInput_;
}

std::shared_ptr<const Matrix> UniVec::getItemOutputMatrix() const {
  return itemOutput_;
}

std::shared_ptr<const Matrix> UniVec::getWordOutputMatrix() const {
  return wordOutput_;
}

void UniVec::saveVectors(const std::string& filename, std::shared_ptr<const Matrix> mat) const {
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        filename + " cannot be opened for saving vectors!");
  }
  // ofs << *mat;
  mat->dump(ofs);
  ofs.close();
}

void UniVec::saveVectors(const std::string& filename) const {
  // saveVectors(filename + "_userInput.vec", userInput_);
  // saveVectors(filename + "_userViewInput.vec", userViewInput_);
  // saveVectors(filename + "_itemInput.vec", itemInput_);
  // saveVectors(filename + "_wordOutput.vec", wordOutput_);
  // saveVectors(filename + "_itemOutput.vec", itemOutput_);
  // saveVectors(filename + "_itemViewOutput.vec", itemViewOutput_);
  size_t Nx = userInput_->rows();
  size_t Ny = userInput_->cols();
  cnpy::npy_save(filename + "_userInput.vec.npy",userInput_->data(),{Nx,Ny},"w");

  Nx = userWordOutput_->rows();
  Ny = userWordOutput_->cols();
  cnpy::npy_save(filename + "_userWordOutput.vec.npy", userWordOutput_->data(),{Nx,Ny},"w");

  Nx = userViewInput_->rows();
  Ny = userViewInput_->cols();
  cnpy::npy_save(filename + "_userViewInput.vec.npy",userViewInput_->data(),{Nx,Ny},"w");

  Nx = itemInput_->rows();
  Ny = itemInput_->cols();
  cnpy::npy_save(filename + "_itemInput.vec.npy",itemInput_->data(),{Nx,Ny},"w");

  Nx = wordOutput_->rows();
  Ny = wordOutput_->cols();
  cnpy::npy_save(filename + "_wordOutput.vec.npy",wordOutput_->data(),{Nx,Ny},"w");

  Nx = itemOutput_->rows();
  Ny = itemOutput_->cols();
  cnpy::npy_save(filename + "_itemOutput.vec.npy",itemOutput_->data(),{Nx,Ny},"w");

  Nx = itemViewOutput_->rows();
  Ny = itemViewOutput_->cols();
  cnpy::npy_save(filename + "_itemViewOutput.vec.npy",itemViewOutput_->data(),{Nx,Ny},"w");
}

bool UniVec::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void UniVec::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void UniVec::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void UniVec::saveModel(const std::string& filename) {
  assert(!quant_);
  std::ofstream ofs(filename, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);

  userInput_->save(ofs);
  itemInput_->save(ofs);
  wordOutput_->save(ofs);
  itemOutput_->save(ofs);
  userWordOutput_->save(ofs);

  ofs.close();
}

void UniVec::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void UniVec::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();

  userInput_ = std::make_shared<Matrix>();
  itemInput_ = std::make_shared<Matrix>();
  wordOutput_ = std::make_shared<Matrix>();
  itemOutput_ = std::make_shared<Matrix>();
  userWordOutput_ = std::make_shared<Matrix>();

  args_->load(in);

  userInput_->load(in);
  itemInput_->load(in);
  wordOutput_->load(in);
  itemOutput_->load(in);
  userWordOutput_->load(in);
  
  model_ = std::make_shared<Model>(itemInput_, userInput_, wordOutput_, itemOutput_, args_, true, 0);

  // model_->setTargetCounts(dict_->getCounts(entry_type::word));
}

void UniVec::printInfo(real progress, real loss, std::ostream& log_stream) {
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double t =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start_)
          .count();
  double lr = args_->lr * (1.0 - progress);
  double wst = 0;

  int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

  if (progress > 0 && t >= 0) {
    progress = progress * 100;
    eta = t * (100 - progress) / progress;
    wst = double(tokenCount_) / t / args_->thread;
  }
  int32_t etah = eta / 3600;
  int32_t etam = (eta % 3600) / 60;

  log_stream << std::fixed;
  log_stream << "Progress: ";
  log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
  log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
  log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
  log_stream << " loss: " << std::setw(9) << std::setprecision(6) << loss;
  log_stream << " ETA: " << std::setw(3) << etah;
  log_stream << "h" << std::setw(2) << etam << "m";
  log_stream << std::flush;
}

void UniVec::regWordModel(Model& itemWordModel, int32_t inputItemIdx, const std::vector<int32_t>& wordVec, real lr) {
  std::vector<int32_t> input {inputItemIdx};
  //const std::vector<int32_t>& wordVec = dataLoader_->item2Word[inputItemIdx];
  for (int i = 0; i < wordVec.size(); i++) {
    itemWordModel.update(input, wordVec, i, lr);
  }
}

void UniVec::trainOnObs(Model& itemWordModel, Model& itemUserModel, Model& userWordModel ,const std::vector<int32_t>& obsVec, real lr) {
  // train on the user-item
  const int32_t userPos = 1;
  const int32_t itemPos = 0;

  if (args_->combine == combine_method::concat) {
    itemUserModel.updateConcat(obsVec, userPos, itemPos, lr);

    // contextual user embedding
    if (!args_->skipUserContext) {
      regWordModel(userWordModel, obsVec[userPos], dataLoader_->user2Word[obsVec[userPos]], lr);
    }

    // contextual item embeddings
    if (!args_->skipContext) {
      if (args_->regOutput) {
        regWordModel(itemWordModel, obsVec[itemPos], dataLoader_->item2Word[obsVec[itemPos]], lr);
      } else {
        std::vector<int32_t> input;
        for (int pos = 0; pos < obsVec.size(); pos++) {
          if (pos == userPos || pos == itemPos) continue;
          int32_t inputItemIdx = obsVec[pos];
          regWordModel(itemWordModel, inputItemIdx, dataLoader_->item2Word[inputItemIdx], lr);
        }
      }
    }
  } else if (args_->combine == combine_method::mean) {

    itemUserModel.updateMean(obsVec, userPos, itemPos, lr);
    
    // contextual user embedding
    if (!args_->skipUserContext) {
      regWordModel(userWordModel, obsVec[userPos], dataLoader_->user2Word[obsVec[userPos]], lr);
    }

    if (args_->skipContext) return;
    regWordModel(itemWordModel, obsVec[itemPos], dataLoader_->item2Word[obsVec[itemPos]], lr);
  } else if (args_->combine == combine_method::meanSum) {
    
    itemUserModel.updateMeanSum(obsVec, userPos, itemPos, lr);

    // contextual user embedding
    if (!args_->skipUserContext) {
      regWordModel(userWordModel, obsVec[userPos], dataLoader_->user2Word[obsVec[userPos]], lr);
    }

    if (args_->skipContext) return;
    regWordModel(itemWordModel, obsVec[itemPos], dataLoader_->item2Word[obsVec[itemPos]], lr);
  }
};

void UniVec::trainOnSubObs(Model& wordModel, Model& model, const std::vector<int32_t>& obsVec, real lr) {
  const int32_t userPos = 1;
  const int32_t itemPos = 0;
  const int32_t subPos = 2;
  assert(obsVec.size() == 3);
  std::vector<int32_t> input;
  input.push_back(obsVec[itemPos]);
  std::vector<int32_t> subVec;
  subVec.push_back(obsVec[subPos]);
  model.update(input, subVec, 0, lr);
  if (!args_->skipContext) regWordModel(wordModel, obsVec[itemPos], dataLoader_->item2Word[obsVec[itemPos]], lr);
}

void UniVec::trainOnSearchObs(Model& model, const std::vector<int32_t>& obsVec, real lr) {
  // item_id, search word1 search word2
  const int32_t itemPos = 0;
  assert(obsVec.size() > 1);
  std::vector<int32_t> input;
  input.push_back(obsVec[itemPos]);
  for(int i = itemPos + 1; i < obsVec.size(); i++) {
    model.update(input, obsVec, i, lr);
  }
}

void UniVec::trainThread(int32_t threadId) {

  Model itemWordModel(itemInput_, userInput_, wordOutput_, itemOutput_, args_, true, threadId);
  Model itemUserModel(itemInput_, userInput_, wordOutput_, itemOutput_, args_, false, threadId);

  itemWordModel.setTargetCounts(dataLoader_->wordCount);

  // An item2word model is the first matrix to the third matrix.
  Model userWordModel(userInput_, userInput_, userWordOutput_, itemOutput_, args_, true, threadId);
  if (!args_->skipUserContext) {
    userWordModel.setTargetCounts(dataLoader_->userWordCount);
  }

  if (!args_->skipTrxData) {
    if (threadId == 0) std::cout << "Generating TRX neg sample table" << std::endl;
    itemUserModel.setTargetCounts(dataLoader_->itemCount);
  }
  
  Model itemUserViewModel(itemInput_, userViewInput_, wordOutput_, itemViewOutput_, args_, false, threadId);
  if (!args_->skipViewData) {
    if (threadId == 0)  std::cout << "Generating VIEW neg sample table" << std::endl;
    itemUserViewModel.setTargetCounts(dataLoader_->itemViewCount);
  }
  
  Model itemSubModel(itemInput_, userInput_, itemInput_, itemOutput_, args_, false, threadId);
  if (!args_->skipSubData) {
    if (threadId == 0)  std::cout << "Generating SUB neg sample table" << std::endl;
    itemSubModel.setTargetCounts(dataLoader_->itemSubCount);
  }

  Model itemSearchModel(itemInput_, userInput_, wordOutput_, itemOutput_, args_, true, threadId);
  if (!args_->skipSearchData) {
    if (threadId == 0)  std::cout << "Generating SEARCH neg sample table" << std::endl;
    itemSearchModel.setTargetCounts(dataLoader_->searchWordCount);
  }

  size_t mSize = 1;

  int64_t localTokenCount = 0;

  int64_t obsIdx = threadId * ntokens / args_->thread;
  int64_t obsViewIdx = threadId * nViewTokens / args_->thread;
  int64_t obsSubIdx = threadId * nSubTokens / args_->thread;
  int64_t obsSearchIdx = threadId * nSearchTokens / args_->thread;

  std::vector<int32_t> line, labels;

  std::cout << "Train start!!" << std::endl;

  while (tokenCount_ < args_->epoch * expectToken) {

    real progress = real(tokenCount_) / (args_->epoch * expectToken);
    real lr = args_->lr * (1.0 - progress);
   
    localTokenCount++;

    obsIdx++;
    obsIdx = obsIdx % ntokens;
    // Anchor - User - Context model with contextual constraints

    if (!args_->skipTrxData) {
      const std::vector<int32_t>& trxObsVec = dataLoader_->allUserHist[obsIdx];
      std::vector<std::vector<int32_t> > trxWinSeq = dataLoader_->computeWindowedOrderedBasket(trxObsVec, 0, args_->ws, args_->shuffleTrxData, rng);
      for (const auto& trx: trxWinSeq) {
        trainOnObs(itemWordModel, itemUserModel, userWordModel, trx, lr);
      }
    }
    
    obsViewIdx++;
    obsViewIdx = obsViewIdx % nViewTokens;
    if (!args_->skipViewData) {
      const std::vector<int32_t>& viewObsVec = dataLoader_->allUserHistView[obsViewIdx];
      std::vector<std::vector<int32_t> > viewWinSeq = dataLoader_->computeWindowedOrderedBasket(viewObsVec, 0, args_->ws, args_->shuffleViewData, rng);
      for (const auto& view: viewWinSeq) {
        trainOnObs(itemWordModel, itemUserViewModel, userWordModel, view, lr);
      }
    }

    // Anchor - Anchor model as a speicial item word model;
    obsSubIdx++;
    obsSubIdx = obsSubIdx % nSubTokens;

    if (!args_->skipSubData) {
      const std::vector<int32_t>& subObsVec = dataLoader_->allUserHistSub[obsSubIdx]; 
      trainOnSubObs(itemWordModel, itemSubModel, subObsVec, lr);
    }

    obsSearchIdx++;
    obsSearchIdx = obsSearchIdx % nSearchTokens;
    if (!args_->skipSearchData) {
      const std::vector<int32_t>& searchObsVec = dataLoader_->allUserHistSearch[obsSearchIdx]; 
      trainOnSearchObs(itemSearchModel, searchObsVec, lr);
    }

    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount_ += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1)
        // loss_ = itemUserModel.getLoss();
        loss_ = itemWordModel.getLoss() + itemUserModel.getLoss() + itemUserViewModel.getLoss() + itemSubModel.getLoss() + itemSearchModel.getLoss() + userWordModel.getLoss();
    }

    // std::cout<<"after update"<< std::endl;
  }
  if (threadId == 0)
    // loss_ = itemUserModel.getLoss();
    loss_ = itemWordModel.getLoss() + itemUserModel.getLoss() + itemUserViewModel.getLoss() + itemSubModel.getLoss() + itemSearchModel.getLoss() + userWordModel.getLoss();
}

void UniVec::loadData(std::shared_ptr<DataLoader> dataLoader) {
  dataLoader_ = dataLoader;

  size_t mSize = 1;
  expectToken = 0;
  // std::cout << "expectToken: " << expectToken << std::endl;
  ntokens = std::max(mSize, dataLoader_->allUserHist.size());
  nViewTokens = std::max(mSize, dataLoader_->allUserHistView.size());
  nSubTokens = std::max(mSize, dataLoader_->allUserHistSub.size());
  nSearchTokens = std::max(mSize, dataLoader_->allUserHistSearch.size());

  if (!args_->skipTrxData) {
    expectToken = ntokens;
  } else if (!args_->skipViewData) {
    expectToken = nViewTokens;
  } else if (!args_->skipSubData) {
    expectToken = nSubTokens;
  } else {
    expectToken = nSearchTokens;
  }
  // std::cout << "expectToken: " << expectToken << std::endl;
}

void UniVec::init(std::shared_ptr<Args> args, std::shared_ptr<DataLoader> dataloader) {
  // args_ = std::make_shared<Args>(args);
  args_ = args;
  loadData(dataloader);
  initMatrix();
}

void UniVec::initMatrix() {
  size_t mSize = 1;
  SizeStats stats = dataLoader_->getSizeStats();

  if (args_->combine == combine_method::concat) {
    userInput_ = std::make_shared<Matrix>(stats.getUserSize(mSize), args_->userDim);
    userViewInput_ = std::make_shared<Matrix>(stats.getUserSize(mSize), args_->userDim);
    userWordOutput_ = std::make_shared<Matrix>(stats.getUserWordSize(mSize), args_->userDim);

    itemInput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim);
    wordOutput_ = std::make_shared<Matrix>(stats.getWordSize(mSize), args_->dim);
    itemOutput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim + args_->userDim);
    itemViewOutput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim + args_->userDim);

  } else {
    assert(args_->userDim == args_->dim);
    userInput_ = std::make_shared<Matrix>(stats.getUserSize(mSize), args_->userDim);
    userViewInput_ = std::make_shared<Matrix>(stats.getUserSize(mSize), args_->userDim);
    userWordOutput_ = std::make_shared<Matrix>(stats.getUserWordSize(mSize), args_->userDim);

    itemInput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim);
    wordOutput_ = std::make_shared<Matrix>(stats.getWordSize(mSize), args_->dim);
    itemOutput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim);
    itemViewOutput_ = std::make_shared<Matrix>(stats.getItemInSize(mSize), args_->dim);
  }

  userInput_->uniform(1.0);
  userViewInput_->uniform(1.0);
  itemInput_->uniform(1.0);
  wordOutput_->uniform(1.0);
  itemOutput_->uniform(1.0);
  itemViewOutput_->uniform(1.0);
}

void UniVec::train(const Args& args) {

  std::cout << "Using comebine method: " << args_->combineToString(args_->combine) << std::endl;

  std::cout << "userInput_ size: " << userInput_->rows() << ", "<< userInput_->cols() << std::endl;
  std::cout << "userViewInput_ size: " << userViewInput_->rows() << ", "<< userViewInput_->cols() << std::endl;
  std::cout << "itemInput_ size: " << itemInput_->rows() << ", "<< itemInput_->cols() << std::endl;
  std::cout << "wordOutput_ size: " << wordOutput_->rows() << ", "<< wordOutput_->cols() << std::endl;
  std::cout << "itemOutput_ size: " << itemOutput_->rows() << ", "<< itemOutput_->cols() << std::endl;
  std::cout << "itemViewOutput_ size: " << itemViewOutput_->rows() << ", "<< itemOutput_->cols() << std::endl;
  
  startThreads();
  model_ = std::make_shared<Model>(itemInput_, userInput_, wordOutput_, itemOutput_, args_, true, 0);
  // model_->setTargetCounts(dict_->getCounts(entry_type::word));

}

void UniVec::startThreads() {
  start_ = std::chrono::steady_clock::now();
  tokenCount_ = 0;
  loss_ = -1;
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args_->thread; i++) {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }

  //  const int64_t ntokens = dataLoader_->allUserHist.size();
  // Same condition as trainThread
  while (tokenCount_ < args_->epoch * expectToken) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (loss_ >= 0 && args_->verbose > 1) {
      real progress = real(tokenCount_) / (args_->epoch * expectToken);
      std::cerr << "\r";
      printInfo(progress, loss_, std::cerr);
    }
  }
  for (int32_t i = 0; i < args_->thread; i++) {
    threads[i].join();
  }
  if (args_->verbose > 0) {
    std::cerr << "\r";
    printInfo(1.0, loss_, std::cerr);
    std::cerr << std::endl;
  }
}

int UniVec::getDimension() const {
  return args_->dim;
}

bool UniVec::isQuant() const {
  return quant_;
}

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r) {
  return l.first > r.first;
}

} // namespace uni_vec
