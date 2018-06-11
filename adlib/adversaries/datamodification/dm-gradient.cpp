/* dm-gradient.cpp
 * C++ implementation of the data-modification gradient calculation.
 * This will be used to speed up the data-modification attack.
 *
 * This code is just meant to be fast - nothing else. It will be poorly
 * designed, leak memory, and be otherwise unpleasant to look at.
 *
 * Matthew Sedam. 2018.
 */

#include <fstream>
#include <future>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

void loadFVS(double **fvs, uint32_t numFeatures) {
    std::ifstream fvsStream("./fvs.txt");
    std::string num;
    uint32_t row = 0;
    uint32_t col = 0;
    while (fvsStream >> num) {
        if (col == numFeatures) {
            ++row;
            col = 0;
        }
        fvs[row][col] = std::strtod(num.c_str(), nullptr);
        ++col;
    }
}

void loadVector(double *vector, const std::string &fileName) {
    std::ifstream stream(fileName);
    std::string num;
    uint32_t idx = 0;
    while (stream >> num) {
        vector[idx] = std::strtod(num.c_str(), nullptr);
        ++idx;
    }
}

void calcPartialfPartialTheta(double **fvs, double *logisticVals, double lda,
                              uint32_t numInstances, uint32_t numFeatures) {

    std::ofstream out("./partial_f_partial_theta.txt");
    out << std::setprecision(15);

    for (uint32_t j = 0; j < numFeatures; ++j) {
        for (uint32_t k = 0; k < numFeatures; ++k) {
            double runningSum = 0.0;
            for (uint32_t i = 0; i < numInstances; ++i) {
                runningSum += fvs[i][k] * fvs[i][j] * logisticVals[i] *
                              (1 - logisticVals[i]);
            }

            if (j == k) {
                runningSum += lda;
            }

            out << runningSum << " ";
        }
        out << std::endl;
    }

}

void calcPartialfPartialD(double **fvs, double *logisticVals, double *theta,
                          double *labels, uint32_t numInstances,
                          uint32_t numFeatures) {

    std::ofstream out("./partial_f_partial_capital_d.txt");
    out << std::setprecision(15);

    uint32_t concurentThreadsSupported = std::thread::hardware_concurrency();
    if (concurentThreadsSupported == 0) {
        concurentThreadsSupported = 4;
    }
    uint32_t chunkSize = numInstances / concurentThreadsSupported;

    std::vector<std::future<std::string>> futures;
    for (uint32_t i = 0; i < concurentThreadsSupported; ++i) {
        uint32_t min = i * chunkSize;
        uint32_t max = (i + 1) * chunkSize;
        if (i == concurentThreadsSupported - 1) {
            max = numInstances;
        }

        futures.push_back(std::async(std::launch::async,
                                     [numFeatures, fvs, theta, logisticVals, labels](uint32_t min,
                                                                                     uint32_t max) {
                                         std::stringstream stream;

                                         for (uint32_t i = min; i < max; ++i) {
                                             double val = logisticVals[i];
                                             double label = labels[i];

                                             for (uint32_t j = 0; j < numFeatures; ++j) {
                                                 for (uint32_t k = 0; k < numFeatures; ++k) {
                                                     double inside = val * theta[k] * fvs[i][j];
                                                     if (j == k) {
                                                         inside -= label;
                                                     }

                                                     stream << (1 - val) * inside << " ";
                                                 }
                                             }
                                             stream << std::endl;
                                         }

                                         return stream.str();
                                     }, min, max));
    }

    for (auto &fut: futures) {
        out << fut.get();
    }
}


int main(int argc, const char *argv[]) {
    // argv = {name, self.lda, self.fvs.shape[0], self.fvs.shape[1]}

    // Create data
    double lda = std::strtod(argv[1], nullptr);
    uint32_t numInstances = (uint32_t) std::stoi(argv[2]);
    uint32_t numFeatures = (uint32_t) std::stoi(argv[3]);

    double *fvs[numInstances];
    for (uint32_t i = 0; i < numInstances; ++i) {
        fvs[i] = new double[numFeatures];
    }

    double logisticVals[numInstances];
    double theta[numFeatures];
    double labels[numInstances];

    loadFVS(fvs, numFeatures);
    loadVector(logisticVals, "./logistic_vals.txt");
    loadVector(theta, "./theta.txt");
    loadVector(labels, "./labels.txt");

    calcPartialfPartialTheta(fvs, logisticVals, lda, numInstances, numFeatures);
    calcPartialfPartialD(fvs, logisticVals, theta, labels, numInstances,
                         numFeatures);
}
