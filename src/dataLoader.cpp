// #include <iomanip>
// #include <iostream>
// #include <sstream>
// #include <queue>
// #include <set>
// #include <stdexcept>
#include <assert.h>

#include "dataLoader.h"

namespace uni_vec {

SizeStats::SizeStats(DataLoader& data) {
      trxItem = data.itemCount.size();
      viewItem = data.itemViewCount.size();
      subItem = data.itemSubCount.size();
      itemDictSize = data.item2Word.size();
      user = data.userPool.size();
      contextWord = data.wordCount.size();
      searchWordMaxIdx = data.searchWordCount.size();
      UserWordSize = data.userWordCount.size();
};

DataLoader::DataLoader(Args* args) {
  args_ = args;

  item2Word = loadContextFromFile(args_->itemWordInput);
  wordCount = computeWordCount(item2Word);

  if (!args_->skipUserContext) {
    user2Word = loadContextFromFile(args_->userWordInput, false);
    userWordCount = computeWordCount(user2Word);
  }
  
  std::cout << "Word Count computed" << std::endl; 

  if (!args_->skipTrxData) {
    allUserHist = loadOrderedBasket(args_->userHistInput);
    std::cout << "basket history (trx) loaded!" << std::endl;
    std::cout << allUserHist.size() << std::endl;
    std::cout << std::endl;

    computeUserPool(userPool, allUserHist, 0);

    int32_t userSize = *(std::max_element(std::begin(userPool), std::end(userPool))) + 1;
    std::cout << "User Pool computed, size:" << userSize << std::endl;

    if (userSize != userPool.size()) {
      std::cout << "The user idx has gaps, the missing user idx from training data will have undefined embeddings." << std::endl;
    }

    auto countStat = computeItemCountAndUserCount(allUserHist, userSize, item2Word.size(), 0);
    userCount = std::move(std::get<0>(countStat));
    itemCount = std::move(std::get<1>(countStat));
    std::cout << "Item(Trx) count and user count computed" << std::endl;
  }

  if (!args_->skipViewData) {
    allUserHistView = loadOrderedBasket(args_->userHistInputView);
    std::cout << "basket history (view) loaded!" << std::endl;
    std::cout << allUserHistView.size() << std::endl;

    computeUserPool(userPool, allUserHistView, 0);
    int32_t userSize = *(std::max_element(std::begin(userPool), std::end(userPool))) + 1;
    std::cout << "User Pool computed, size:" << userSize << std::endl;

    auto countStat = computeItemCountAndUserCount(allUserHistView, userSize, item2Word.size(), 0);
    userViewCount = std::move(std::get<0>(countStat));
    itemViewCount = std::move(std::get<1>(countStat));
    std::cout << "Item(View) count and user count computed" << std::endl;
  }

  if (!args_->skipSubData) {
     allUserHistSub = loadTsvFromFile(args_->userHistInputSub, 1);
    std::cout << "basket history (sub) loaded!" << std::endl;
    std::cout << allUserHistSub.size() << std::endl;
    itemSubCount = computeSubItemCount(allUserHistSub, 1);
  }

  if (!args_->skipSearchData) {
    allUserHistSearch = loadTsvFromFile(args_->userHistInputSearch, -1);
    std::cout << "basket history (search) loaded!" << std::endl;
    std::cout << allUserHistSearch.size() << std::endl;
    searchWordCount = computeSearchWordCount();
  }
}

SizeStats DataLoader::getSizeStats() {
  return SizeStats(*this);
}

int2VecOfInt DataLoader::loadContextFromFile(const std::string& contextFileName, bool checkIdxGap) {
  // read from txt to underded_map<int, vector<int>>
  std::ifstream ifs(contextFileName);
  int32_t maxItemIdx = 0;
  int32_t maxWordIdx = 0;
  std::string line;
  int2VecOfInt item2Word;

  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    bool first = true;
    int32_t itemIdx;
    int32_t bufferInt;
    while(iss >> bufferInt) {
      assert(bufferInt >= 0);
      if (first) {
        itemIdx = bufferInt;
        maxItemIdx = std::max(maxItemIdx, itemIdx);
        item2Word[itemIdx] = std::vector<int32_t>(); 
        first = false;
      } else {
        maxWordIdx = std::max(maxWordIdx, bufferInt);
        item2Word[itemIdx].push_back(bufferInt);
      }
    }
  }
  ifs.close();

  // check that the input item idx have no gaps.
  if (checkIdxGap) {
    if (maxItemIdx != item2Word.size() - 1) {
    throw std::invalid_argument("The input item idx has gap! maxItemIdx is:" 
    + std::to_string(maxItemIdx) + "size is: " + std::to_string(item2Word.size()));
    }
  }

  return item2Word;
}

void DataLoader::computeUserPool(std::set<int32_t>& pool, const std::vector<std::vector<int32_t> >& history, int32_t userPos) {
  for (const auto &vec: history) {
    pool.insert(vec[userPos]);
  }
}

std::vector<std::string> parseToVec(const std::string& line, const std::string& delim) {
  std::vector<std::string> res;
  std::string token;
  auto start = 0U;
  auto end = line.find(delim);
  while (end != std::string::npos) {
    token = line.substr(start, end - start);
    res.push_back(token);
    start = end + delim.length();
    end = line.find(delim, start);
  }
  token = line.substr(start, end);
  res.push_back(token);
  return res;
}

std::vector<std::vector<int32_t> > DataLoader::loadOrderedBasket(const std::string& fileName) {
  std::ifstream ifs(fileName);
  std::string line;
  std::vector<std::vector<int32_t> > userHist;

  const std::string tabDeli = "\t";
  const std::string commaDeli = ",";

  int64_t lineCount = 0;
  int64_t errCount = 0;
  while (std::getline(ifs, line)) {
    lineCount += 1;
    std::vector<int32_t> currLine;
    std::vector<std::string> parsedSubStr;
    // Parse the line into three columns
    parsedSubStr = parseToVec(line, tabDeli);
    if (parsedSubStr.size() != 3) {
      throw std::invalid_argument("Each line must has exeactly three columns, line: " + 
      std::to_string(lineCount) + " has: " + std::to_string(parsedSubStr.size()));
    }

    std::vector<std::string> parsedToken;
    // Parsing the first user idx column
    parsedToken = parseToVec(parsedSubStr[0], commaDeli);
    if (parsedToken.size() != 1) {
      throw std::invalid_argument("User idx column should not has more than one entry");
    }
    currLine.push_back(std::stoi(parsedToken[0]));

    // Parsing the time stamp column
    std::vector<double> timeStamps;
    parsedToken = parseToVec(parsedSubStr[1], commaDeli);
    for (const std::string& val: parsedToken) {
        timeStamps.push_back(std::stod(val));
    }
    // Parsing the item column
    std::vector<int32_t> itemVec;
    parsedToken = parseToVec(parsedSubStr[2], commaDeli);
    for (const std::string& val: parsedToken) {
        itemVec.push_back(std::stoi(val));
    }

    if (timeStamps.size() != itemVec.size()) {
      throw std::invalid_argument("Input timestamps and item sequence should have the same length!");
    }

    // Recorder and push the items to the observations;
    std::vector<std::pair<double, int32_t>> recordVec;
    for (int i = 0; i < timeStamps.size(); i++) {
      recordVec.push_back(std::make_pair(timeStamps[i], itemVec[i]));
    }
    std::sort(recordVec.begin(), recordVec.end(), 
    [](const std::pair<double, int32_t> &left, const std::pair<double, int32_t> &right) {
      return left.first < right.first;
    });
    for (int i = 0; i < recordVec.size(); i++) {
      currLine.push_back(recordVec[i].second);
    }

    if (currLine.size() <= 2) {
      errCount += 1;
      continue;
    } else {
      userHist.push_back(currLine);
    }
  }
  std::cout << "Number of orders ignored: " << errCount << ", number of lines in total: " << lineCount << std::endl;
  return userHist;
}

std::vector<std::vector<int32_t> > DataLoader::loadTsvFromFile(const std::string& fileName, int32_t userPos) {

  std::ifstream ifs(fileName);
  std::string line;
  int32_t bufferInt;
  const std::string tabDeli = "\t";

  std::vector<std::vector<int32_t> > userHist;
  while (std::getline(ifs, line)) {
    std::vector<int32_t> currUserHist;
    int32_t pos = 0;
    std::vector<std::string> parsedSubStr;
    parsedSubStr = parseToVec(line, tabDeli);
    // first is cidx. rest all item idx
    for (const std::string& val : parsedSubStr) {
      bufferInt = std::stoi(val);
      assert(bufferInt >= 0);
      // if (pos != userPos && item2Word.find(bufferInt) == item2Word.end()) {
      //   throw std::invalid_argument("The history file contains item idx which is not in the item2Word file!");
      // }
      currUserHist.push_back(bufferInt);
      pos++;
    }
    // if there is user then size >= 3, otherwise size >= 2
    assert(currUserHist.size() >= 2 + int(userPos >= 0));
    userHist.push_back(currUserHist);
  }
  ifs.close();
  return userHist;
}

std::vector<std::vector<int32_t> > DataLoader::computeWindowedOrderedBasket(const std::vector<int32_t>& userBasket, int32_t inputUserPos, int ws, bool shuffle, std::default_random_engine rng) {
  std::vector<std::vector<int32_t> > res;
  std::vector<int32_t> curr;

  int32_t userIdx;
  for (int i = 0; i < userBasket.size(); i++) {
    if (i == inputUserPos) {
      userIdx = userBasket[i];
    } else {
      curr.push_back(userBasket[i]);
    }
  }
  if (shuffle) {
    std::shuffle(std::begin(curr), std::end(curr), rng);
  }
  assert(curr.size() > 1);
  for (int i = 1; i < curr.size(); i ++) {
    int k = std::max(0, i - ws);
    std::vector<int32_t> newLine;
    newLine.push_back(curr[i]);
    newLine.push_back(userIdx);
    for (int j = k; j < i; j++) {
      newLine.push_back(curr[j]);
    }
    assert(newLine.size() > 2);
    res.push_back(newLine);
  }

  return res;
}

std::vector<int64_t> DataLoader::computeSubItemCount(const std::vector<std::vector<int32_t>>& hist, int32_t userPos) {
  std::vector<int64_t> count;
  count.resize(item2Word.size());
  std::fill(count.begin(), count.end(), 1);
  for (const auto& vec: hist) {
    for (int i = 0; i < vec.size(); i++) {
      if (i == userPos) continue;
      count[vec[i]] += 1;
    }
  }
  return count;
}

std::vector<int64_t> DataLoader::computeWordCount(const int2VecOfInt& item2Word) {
  std::set<int32_t> wordSet;
  std::vector<int64_t> wordCount;

  for (const auto& pair: item2Word) {
    for (int32_t wordIdx: pair.second) {
      wordSet.insert(wordIdx);
    }
  }; 
  // The index should make sure all real context word have small index values.
  int32_t minWordIdx = *(std::min_element(std::begin(wordSet), std::end(wordSet)));
  int32_t maxWordIdx = *(std::max_element(std::begin(wordSet), std::end(wordSet)));
  assert(maxWordIdx - minWordIdx + 1 == wordSet.size());

  wordCount.resize(maxWordIdx + 1);
  std::fill(wordCount.begin(), wordCount.end(), 1);
  for (const auto& pair: item2Word) {
    for (int32_t wordIdx: pair.second) {
      wordCount[wordIdx] += 1;
    }
  };

  // check that the word idx have no gaps.
  for (int32_t i = 0; i < wordCount.size(); i++) {
    if (wordCount[i] == 0) {
      throw std::invalid_argument("The input word idx has gap at " + std::to_string(i));
    }
  }
  return wordCount;
}

std::vector<int64_t> DataLoader::computeSearchWordCount() {
  int32_t maxWordIdx = 0;
  std::vector<int64_t> wordCount;

  for (const auto& vec: allUserHistSearch) {
    for (int i = 1; i < vec.size(); i++) {
      maxWordIdx = std::max(maxWordIdx, vec[i]);
    }
  };

  wordCount.resize(maxWordIdx + 1);
  std::fill(wordCount.begin(), wordCount.end(), 1);

  for (const auto& vec: allUserHistSearch) {
    for (int i = 1; i < vec.size(); i++) {
      wordCount[vec[i]] += 1;
    }
  };

  return wordCount;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> DataLoader::computeItemCountAndUserCount(const std::vector<std::vector<int32_t>>& hist, int32_t userSize, int32_t itemSize, int32_t userPos) {
  std::vector<int64_t> uCount;
  uCount.resize(userSize);
  std::fill(uCount.begin(), uCount.end(), 1);

  std::vector<int64_t> iCount;
  iCount.resize(itemSize);
  std::fill(iCount.begin(), iCount.end(), 1);

  for (const auto& vec: hist) {
    for (int i = 0; i < vec.size(); i++) {
      if (i == userPos) {
        assert(vec[i] < userSize);
        uCount[vec[i]] += 1;
      } else {
        assert(vec[i] < itemSize);
        iCount[vec[i]] += 1;
      }
    }
  }
  return std::make_tuple(uCount, iCount);
}

void DataLoader::computeItemViewCount() {
  itemViewCount.resize(item2Word.size());
  std::fill(itemViewCount.begin(), itemViewCount.end(), 1);

  for (const auto& vec: allUserHistView) {
    for (int i = 0; i < vec.size(); i++) {
      if (i == 0) {
        continue;
      } else {
        itemViewCount[vec[i]] += 1;
      }
    }
  }
}


}
