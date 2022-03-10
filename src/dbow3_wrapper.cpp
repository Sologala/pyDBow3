
#include "BowVector.h"
#include "FeatureVector.h"

#include <iostream>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>
#include <tuple>
namespace py = pybind11;

#include "DBoW3.h"
#include "string"
#include <vector>

#include "ndarray_converter.h"
using namespace pybind11::literals;
using namespace std;
string version() { return "0.0.1"; }

/*
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    DBoW3::Vocabulary voc(k, L, weight, score);
 */

static std::map<std::string, DBoW3::WeightingType> all_weight_method{
    {"TF_IDF", DBoW3::WeightingType::TF_IDF},
    {"BINARY", DBoW3::WeightingType::BINARY},
    {"IDF", DBoW3::WeightingType::IDF},
    {"TF", DBoW3::WeightingType::TF}};
static std::map<std::string, DBoW3::ScoringType> all_score_method{
    {"L1_NORM", DBoW3::ScoringType::L1_NORM},
    {"L2_NORM", DBoW3::ScoringType::L2_NORM},
    {"DOT_PRODUCT", DBoW3::ScoringType::DOT_PRODUCT},
    {"BHATTACHARYYA", DBoW3::ScoringType::BHATTACHARYYA},
    {"KL", DBoW3::ScoringType::KL}};
static std::map<DBoW3::WeightingType, std::string> rvt_all_weight_method{
    {DBoW3::WeightingType::TF_IDF, "TF_IDF"},
    {DBoW3::WeightingType::BINARY, "BINARY"},
    {DBoW3::WeightingType::IDF, "IDF"},
    {DBoW3::WeightingType::TF, "TF"}};
static std::map<DBoW3::ScoringType, std::string> rvt_all_score_method{
    {DBoW3::ScoringType::L1_NORM, "L1_NORM"},
    {DBoW3::ScoringType::L2_NORM, "L2_NORM"},
    {DBoW3::ScoringType::DOT_PRODUCT, "DOT_PRODUCT"},
    {DBoW3::ScoringType::BHATTACHARYYA, "BHATTACHARYYA"},
    {DBoW3::ScoringType::KL, "KL"}};

class Vocabulary {
public:
  Vocabulary(int k = 10, int L = 6, const std::string &weight_method = "TF_IDF",
             const std::string &score_method = "L1_NORM", bool verbose = true) {
    assert(all_weight_method.count(weight_method));
    assert(all_score_method.count(score_method));

    voc = new DBoW3::Vocabulary(k, L, all_weight_method[weight_method],
                                all_score_method[score_method]);
  }

  ~Vocabulary() {
    if (_verbose)
      std::cout << "Entering destructor" << std::endl;
    voc->clear();
    delete voc;
    if (_verbose)
      std::cout << "Exiting destructor" << std::endl;
  }

  void create(cv::Mat &training_feat_vec) {
    int N = training_feat_vec.rows;
    int feature_dim = training_feat_vec.cols;
    std::vector<cv::Mat> tf(N);
    for (int i = 0, sz = N; i < sz; i++) {
      tf[i] = training_feat_vec.row(i);
    }
    voc->create(tf);
  }

  void clear() { voc->clear(); }

  void readFromFile(const std::string &path) { voc->load(path); }

  void saveToFile(const std::string &path, bool binary_compress = true) {
    voc->save(path, binary_compress);
  }

  std::tuple<std::map<unsigned int, double>,
             std::map<unsigned int, std::vector<unsigned int>>>
  transform(const std::vector<cv::Mat> &training_feat_vec, int level) {
    DBoW3::BowVector bv;
    DBoW3::FeatureVector fv;
    voc->transform(training_feat_vec, bv, fv, level);

    std::map<unsigned int, std::vector<unsigned int>> ret2 =
        static_cast<std::map<unsigned int, std::vector<unsigned int>>>(fv);

    return std::make_tuple(bv, ret2);
  }

  bool _verbose = false;
  cv::Mat getWord(uint32_t word_id) { return voc->getWord(word_id); }
  std::map<uint32_t, uint32_t> wordId2NodeId() {
    return voc->getNodeId2WordId();
  }
  uint32_t getDepth() { return voc->getDepthLevels(); }
  uint32_t getDescriptorSize() { return voc->getDescritorSize(); }
  uint32_t getWordSize() { return voc->getWordSize(); }

  std::string log() {
    char str_buffer[256] = {};
    sprintf(str_buffer,
            "nwords: %d\ndepths: %d\ndescriptor size %d \nK: %d \nweight "
            "method: %s\nscore method: %s\n",
            getWordSize(), getDepth(), getDescriptorSize(),
            voc->getBranchingFactor(),
            rvt_all_weight_method[voc->getWeightingType()].c_str(),
            rvt_all_score_method[voc->getScoringType()].c_str());
    return std::string(str_buffer);
  }
  DBoW3::Vocabulary *voc;
};

PYBIND11_MODULE(pyDBow3, m) {
  NDArrayConverter::init_numpy();
  m.doc() = "pybind11 of fbow"; // optional module docstring
  m.def("__version__", &version, "get the version of fbow");
  py::class_<Vocabulary>(m, "Vocabulary")
      .def(py::init<int, int, std::string, std::string, bool>(), "K"_a = 10,
           "L"_a = 6, "weight_method"_a = "TF_IDF",
           "score_method"_a = "L1_NORM", "verbose"_a = true)
      //   .def("test", &Vocabulary::temp, "mat"_a)
      .def("create", &Vocabulary::create, "features"_a)
      .def("clear", &Vocabulary::clear)
      .def("readFromFile", &Vocabulary::readFromFile, "path"_a)
      .def("saveToFile", &Vocabulary::saveToFile, "path"_a,
           "binary_compress"_a = true)
      .def("transform", &Vocabulary::transform, "feature"_a, "level"_a)
      .def("getWordSize", &Vocabulary::getWordSize)
      .def("getDescriptorSize", &Vocabulary::getDescriptorSize)
      .def("getDepth", &Vocabulary::getDepth)
      .def("getWord", &Vocabulary::getWord, "word_id"_a)
      .def("wordIdd2NodeId", &Vocabulary::wordId2NodeId)
      .def("__str__", &Vocabulary::log)
      .def("clear", &Vocabulary::clear);
}
