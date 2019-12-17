#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <queue>
#include <set>
#include <stdexcept>
#include <random>
#include <algorithm>

#include "real.h"
#include "args.h"
#include "utils.h"

namespace uni_vec {

class DataLoader;

class SizeStats {
  /*Record the statistics and logics to compute the size for constructing the embedding matrix */
  public:
    SizeStats(DataLoader& data);
    int64_t getItemInSize(int64_t minSize) {
      return itemDictSize;//std::max(minSize, std::max(std::max(trxItem, viewItem), subItem));
    }

    int64_t getUserSize(int64_t minSize) {
      return std::max(minSize, user);
    }
    
    int64_t getWordSize(int64_t minSize) {
      return std::max(minSize, std::max(contextWord, searchWordMaxIdx));
    }

    int64_t getUserWordSize(int64_t minSize) {
       return std::max(minSize, UserWordSize);
    }
    
  private:
    int64_t itemDictSize;
    int64_t trxItem;
    int64_t viewItem;
    int64_t subItem;
    int64_t user;
    int64_t searchWordMaxIdx;
    int64_t contextWord;
    int64_t UserWordSize;
};

class DataLoader {
  public:
    int2VecOfInt item2Word;
    int2VecOfInt user2Word;
    
    std::vector<std::vector<int32_t> > allUserHist; // trx
    std::vector<std::vector<int32_t> > allUserHistView; // view
    std::vector<std::vector<int32_t> > allUserHistSub; // sub
    std::vector<std::vector<int32_t> > allUserHistSearch; // search

    std::set<int32_t> userPool;
    std::set<int32_t> itemOutputPool;
    std::set<int32_t> itemInputPool;

    std::vector<int64_t> wordCount;
    std::vector<int64_t> userWordCount;

    std::vector<int64_t> searchWordCount;
    std::vector<int64_t> userCount;
    std::vector<int64_t> userViewCount;
    std::vector<int64_t> itemCount;
    std::vector<int64_t> itemViewCount;
    std::vector<int64_t> itemSubCount;
    
    void computeUserPool(std::set<int32_t>&, const std::vector<std::vector<int32_t> >&, int32_t);

    int2VecOfInt loadContextFromFile(const std::string&, bool checkIdxGap=true);

    std::vector<std::vector<int32_t> > loadOrderedBasket(const std::string&);
    std::vector<std::vector<int32_t> > loadTsvFromFile(const std::string&, int32_t);
    std::vector<std::vector<int32_t> > computeWindowedOrderedBasket(const std::vector<int32_t>&, int32_t, int32_t, bool, std::default_random_engine);

    std::tuple<std::vector<int64_t>, std::vector<int64_t>> computeItemCountAndUserCount(const std::vector<std::vector<int32_t>>& hist, int32_t userSize, int32_t itemSize, int32_t userPos);
    void computeItemViewCount();
    std::vector<int64_t> computeSubItemCount(const std::vector<std::vector<int32_t>>&, int32_t);
    std::vector<int64_t> computeWordCount(const int2VecOfInt&);
    std::vector<int64_t> computeSearchWordCount();

    DataLoader(Args* args);
    
    int2VecOfInt& getItem2Word();
    std::vector<std::vector<int32_t> >& getAllUserHist();
    std::set<int32_t>& getUserPool();

    // std::set<int32_t>& getItemOutputPool();
    // std::set<int32_t>& getItemInputPool();

    SizeStats getSizeStats();
    std::vector<int64_t>& getWordCount();
    std::vector<int64_t>& getUserCount();
    std::vector<int64_t>& getItemCount();
    std::vector<int64_t>& getItemViewCount();

  protected:
    Args* args_;
};
}