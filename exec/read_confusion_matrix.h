#ifndef _READ_CONFUSION_MATRIX_H_
#define _READ_CONFUSION_MATRIX_H_

#include "csv.h"

template<size_t N>
Eigen::Matrix<double, N, N> ReadConfusionMatrix(std::string& file_name) {
  Eigen::Matrix<double, N, N> out;

  std::ifstream filestream(file_name);
  size_t j = 0;
  for(CSVIterator loop(filestream); loop != CSVIterator(); ++loop) {
    for(size_t k =0; k <(*loop).size(); k++) {
      out(j,k) = std::stod((*loop)[k]);
    }
    j++;
  }
  return out;
}

#endif  // _READ_CONFUSIONMATRIX_H_
