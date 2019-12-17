/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "args.h"

#include <stdlib.h>

#include <iostream>
#include <stdexcept>

namespace uni_vec {

Args::Args() {
  lr = 0.05;
  dim = 100;
  userDim = -1;
  ws = 5;
  epoch = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  combine = combine_method::concat;
  bucket = 2000000;
  minn = 3;
  maxn = 6;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;
  pretrainedVectors = "";
  saveOutput = false;
  useConcat = false;
  regOutput = false;
  quasiAtten = false;
  skipUserContext = false;
  
  skipTrxData = false;
  skipViewData = false;
  skipSubData = false;
  skipSearchData = false;
 
  shuffleTrxData = true;
  shuffleViewData = true;

  qout = false;
  retrain = false;
  qnorm = false;
  skipContext = false;
  cutoff = 0;
  dsub = 2;
}

std::string Args::lossToString(loss_name ln) const {
  switch (ln) {
    case loss_name::hs:
      return "hs";
    case loss_name::ns:
      return "ns";
    case loss_name::softmax:
      return "softmax";
    case loss_name::ova:
      return "one-vs-all";
  }
  return "Unknown loss!"; // should never happen
}

std::string Args::boolToString(bool b) const {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

std::string Args::modelToString(model_name mn) const {
  switch (mn) {
    case model_name::cbow:
      return "cbow";
    case model_name::sg:
      return "sg";
    case model_name::sup:
      return "sup";
  }
  return "Unknown model name!"; // should never happen
}

std::string Args::combineToString(combine_method cm) const {
  switch (cm) {
    case combine_method::concat:
      return "concat";
    case combine_method::mean:
      return "mean";
    case combine_method::meanSum:
      return "meanSum";
  }
  return "Unknown model name!"; // should never happen 
}

void Args::parseArgs(const std::vector<std::string>& args) {
  std::string command(args[1]);

  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    maxn = 0;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      std::cerr << args[ai] << std::endl;

      printHelp();
      exit(EXIT_FAILURE);
    }
    try {
      if (args[ai] == "-h") {
        std::cerr << "Here is the help! Usage:" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      } else if (args[ai] == "-input") {
        input = std::string(args.at(ai + 1));
      } else if (args[ai] == "-itemWordInput") {
        itemWordInput = std::string(args.at(ai + 1));
      } else if (args[ai] == "-userWordInput") {
        userWordInput = std::string(args.at(ai + 1));
      } else if (args[ai] == "-userHistInput") {
        userHistInput = std::string(args.at(ai + 1));
      } else if (args[ai] == "-userHistInputView") {
        userHistInputView = std::string(args.at(ai + 1));
      } else if (args[ai] == "-userHistInputSub") {
        userHistInputSub = std::string(args.at(ai + 1));
      } else if (args[ai] == "-userHistInputSearch") {
        userHistInputSearch = std::string(args.at(ai + 1));
      } else if (args[ai] == "-output") {
        output = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lr") {
        lr = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-lrUpdateRate") {
        lrUpdateRate = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dim") {
        dim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-userDim") {
        userDim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-ws") {
        ws = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-epoch") {
        epoch = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCount") {
        minCount = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCountLabel") {
        minCountLabel = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-neg") {
        neg = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordNgrams") {
        wordNgrams = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-loss") {
        if (args.at(ai + 1) == "hs") {
          loss = loss_name::hs;
        } else if (args.at(ai + 1) == "ns") {
          loss = loss_name::ns;
        } else if (args.at(ai + 1) == "softmax") {
          loss = loss_name::softmax;
        } else if (
            args.at(ai + 1) == "one-vs-all" || args.at(ai + 1) == "ova") {
          loss = loss_name::ova;
        } else {
          std::cerr << "Unknown loss: " << args.at(ai + 1) << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      } else if (args[ai] == "-combineMethod") {
        if (args.at(ai + 1) == "concat") {
          combine = combine_method::concat;
        } else if (args.at(ai + 1) == "mean") {
          combine = combine_method::mean;
        } else if (args.at(ai + 1) == "meanSum") {
          combine = combine_method::meanSum;
        } else {
          std::cerr << "Unknown combine method: " << args.at(ai + 1) << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      } else if (args[ai] == "-bucket") {
        bucket = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minn") {
        minn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxn") {
        maxn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-thread") {
        thread = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-t") {
        t = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-label") {
        label = std::string(args.at(ai + 1));
      } else if (args[ai] == "-verbose") {
        verbose = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-pretrainedVectors") {
        pretrainedVectors = std::string(args.at(ai + 1));
      } else if (args[ai] == "-saveOutput") {
        saveOutput = true;
        ai--;
      } else if (args[ai] == "-skipContext") {
        skipContext = true;
        ai--;
      } else if (args[ai] == "-skipUserContext") {
        skipUserContext = true;
        ai--;
      } else if (args[ai] == "-skipTrxData") {
        skipTrxData = true;
        ai--;
      } else if (args[ai] == "-skipViewData") {
        skipViewData = true;
        ai--;
      } else if (args[ai] == "-skipSubData") {
        skipSubData = true;
        ai--;
      } else if (args[ai] == "-skipSearchData") {
        skipSearchData = true;
        ai--;
      } else if (args[ai] == "-regOutput") {
        regOutput = true;
        ai--;
      } else if (args[ai] == "-useConcat") {
        useConcat = true;
        ai--;
      } else if (args[ai] == "-quasiAtten") {
        quasiAtten = true;
        ai--;
      } else if (args[ai] == "-qnorm") {
        qnorm = true;
        ai--;
      } else if (args[ai] == "-retrain") {
        retrain = true;
        ai--;
      } else if (args[ai] == "-qout") {
        qout = true;
        ai--;
      } else if (args[ai] == "-cutoff") {
        cutoff = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dsub") {
        dsub = std::stoi(args.at(ai + 1));
      } else {
        std::cerr << "Unknown argument: " << args[ai] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } catch (std::out_of_range) {
      std::cerr << args[ai] << " is missing an argument" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
   
  // overide the skip data options to true if such data files are not provided.
  if(this->userWordInput.empty()) {
    skipUserContext = true;
  }
  if (this->userHistInput.empty()) {
    skipTrxData = true;
  }
  if (this->userHistInputView.empty()) {
    skipViewData = true;
  }
  if (this->userHistInputSearch.empty()) {
    skipSearchData = true;
  }
  if (this->userHistInputSub.empty()) {
    skipSubData = true;
  }


  if (output.empty() || itemWordInput.empty() || userHistInput.empty()) {
    std::cerr << "Empty input word-item or user hist or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }

  // check user embeddings dim 
  if (combine != combine_method::concat) {
    if (userDim != -1 && userDim != dim) {
      std::cerr << "If use mean for user item embeddings, then don't provide userDim! Or make sure they are equal" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else {
      userDim = dim;
    }
  } else if (combine != combine_method::concat) {
    if (userDim == -1) {
      userDim = dim;
      // std::cerr << "userDim not provided for concate mode of user-item embeddings" << std::endl;
    }
  }

  if (wordNgrams <= 1 && maxn == 0) {
    bucket = 0;
  }

  if (skipViewData && skipTrxData && skipSubData && skipSearchData) {
    std::cerr << "Can not skip all data, at least one task data from trx, view, sub or search need to be used" << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
  printQuantizationHelp();
}

void Args::printBasicHelp() {
  std::cerr << "\nThe following arguments are mandatory:\n"
            << "  -itemWordInput      item training file path\n"
            << "  -userHistInput      user hist training file path\n"
            << "  -output             output file path\n"
            << "\nThe following arguments are optional:\n"
            << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printTrainingHelp() {
  std::cerr
      << "\nThe following arguments for training are optional:\n"
      << "  -lr                 learning rate [" << lr << "]\n"
      << "  -lrUpdateRate       change the rate of updates for the learning rate ["
      << lrUpdateRate << "]\n"
      << "  -dim                size of word vectors [" << dim << "]\n"
      << "  -ws                 size of the context window [" << ws << "]\n"
      << "  -epoch              number of epochs [" << epoch << "]\n"
      << "  -neg                number of negatives sampled [" << neg << "]\n"
      << "  -thread             number of threads [" << thread << "]\n"
      << "  -saveOutput         whether output params should be saved ["
      << boolToString(saveOutput) << "]\n"
      << "  -userWordInput      location of user context [" << userWordInput << "]\n"
      << "  -userHistInputView  location of the user view history [" << userHistInputView << "]\n"

      << "  -skipContext        not consider item conext for training [" << skipContext << "]\n"
      << "  -skipUserContext    not consider user conext for training [" << skipUserContext << "]\n"
      << "  -regOutput          consider conext constraints on the target items in stead of condition items [" << regOutput << "]\n";
}


void Args::printDictionaryHelp() {
  std::cerr << "\nNot implemented, please ignore\n";
}

void Args::printQuantizationHelp() {
  std::cerr
      << "\nNot implemented, please ignore\n";
}

void Args::save(std::ostream& out) {
  out.write((char*)&(dim), sizeof(int));
  out.write((char*)&(ws), sizeof(int));
  out.write((char*)&(epoch), sizeof(int));
  out.write((char*)&(minCount), sizeof(int));
  out.write((char*)&(neg), sizeof(int));
  out.write((char*)&(wordNgrams), sizeof(int));
  out.write((char*)&(loss), sizeof(loss_name));
  out.write((char*)&(model), sizeof(model_name));
  out.write((char*)&(bucket), sizeof(int));
  out.write((char*)&(minn), sizeof(int));
  out.write((char*)&(maxn), sizeof(int));
  out.write((char*)&(lrUpdateRate), sizeof(int));
  out.write((char*)&(t), sizeof(double));
}

void Args::load(std::istream& in) {
  in.read((char*)&(dim), sizeof(int));
  in.read((char*)&(ws), sizeof(int));
  in.read((char*)&(epoch), sizeof(int));
  in.read((char*)&(minCount), sizeof(int));
  in.read((char*)&(neg), sizeof(int));
  in.read((char*)&(wordNgrams), sizeof(int));
  in.read((char*)&(loss), sizeof(loss_name));
  in.read((char*)&(model), sizeof(model_name));
  in.read((char*)&(bucket), sizeof(int));
  in.read((char*)&(minn), sizeof(int));
  in.read((char*)&(maxn), sizeof(int));
  in.read((char*)&(lrUpdateRate), sizeof(int));
  in.read((char*)&(t), sizeof(double));
}

void Args::dump(std::ostream& out) const {
  out << "dim"
      << " " << dim << std::endl;
  out << "ws"
      << " " << ws << std::endl;
  out << "epoch"
      << " " << epoch << std::endl;
  out << "minCount"
      << " " << minCount << std::endl;
  out << "neg"
      << " " << neg << std::endl;
  out << "wordNgrams"
      << " " << wordNgrams << std::endl;
  out << "loss"
      << " " << lossToString(loss) << std::endl;
  out << "model"
      << " " << modelToString(model) << std::endl;
  out << "bucket"
      << " " << bucket << std::endl;
  out << "minn"
      << " " << minn << std::endl;
  out << "maxn"
      << " " << maxn << std::endl;
  out << "lrUpdateRate"
      << " " << lrUpdateRate << std::endl;
  out << "t"
      << " " << t << std::endl;
}

} // namespace uni_vec
