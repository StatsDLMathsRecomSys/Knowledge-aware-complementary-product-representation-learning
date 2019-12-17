/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <sstream>
#include <queue>
#include <set>
#include <stdexcept>

#include "args.h"
#include "utils.h"
#include "uniVec.h"
#include "dataLoader.h"

using namespace uni_vec;

void printUsage() {
  std::cerr
      << "usage: uni_vec <command> <args>\n\n"
      << "The commands supported by uni_vec are:\n\n"
      << "Sorry, there is currently no commands supported for this version LoL"
      << std::endl;
}

void printQuantizeUsage() {
  std::cerr << "usage: uni_vec quantize <args>" << std::endl;
}

void printPredictUsage() {
  std::cerr
      << "usage: fasttext predict[-prob] <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printTestLabelUsage() {
  std::cerr
      << "usage: fasttext test-label <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr << "usage: fasttext print-word-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr << "usage: fasttext print-sentence-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr << "usage: fasttext print-ngrams <model> <word>\n\n"
            << "  <model>      model filename\n"
            << "  <word>       word to print\n"
            << std::endl;
}

void printNNUsage() {
  std::cout << "usage: fasttext nn <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printAnalogiesUsage() {
  std::cout << "usage: fasttext analogies <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printDumpUsage() {
  std::cout << "usage: fasttext dump <model> <option>\n\n"
            << "  <model>      model filename\n"
            << "  <option>     option from args,dict,input,output" << std::endl;
}

void train(const std::vector<std::string> args) {
  Args a = Args();
  a.parseArgs(args);
  UniVec uniVec;

  std::string outputFileName(a.output + ".bin");
  std::ofstream ofs(outputFileName);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        outputFileName + " cannot be opened for saving.");
  }
  ofs.close();

  std::shared_ptr<Args> argsPtr = std::make_shared<Args>(a);
  std::shared_ptr<DataLoader> dataLoader = std::make_shared<DataLoader>(argsPtr.get());
  std::cout << "Data loaded!" << std::endl;
  uniVec.init(argsPtr, dataLoader);
  std::cout << "Model intialized!" << std::endl;

  uniVec.train(a);
  
  // uniVec.saveModel(outputFileName);
  std::cout <<  uniVec.getUserInputMatrix()->cols() << std::endl;
  uniVec.saveVectors(a.output + ".vec");
}

void dump(const std::vector<std::string>& args) {
  if (args.size() < 4) {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }

  std::string modelPath = args[2];
  std::string option = args[3];

  UniVec uniVec;
  uniVec.loadModel(modelPath);
  if (option == "args") {
    uniVec.getArgs().dump(std::cout);
  } else if (option == "user_input") {
    uniVec.getUserInputMatrix()->dump(std::cout);
  } else if (option == "item_input") {
    uniVec.getItemInputMatrix()->dump(std::cout);
  } else if (option == "word_output") {
    uniVec.getWordOutputMatrix()->dump(std::cout);
  } else if (option == "item_output") {
    uniVec.getItemOutputMatrix()->dump(std::cout);
  } else {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {

  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "test") {
  ;
  } else if (command == "train") {
    train(args);

  } else if (command == "dump") {
    dump(args);

  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }

  return 0;
}
