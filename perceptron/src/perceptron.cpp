// C
#include <assert.h>
// C++
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include <filesystem>
namespace os = std::filesystem;
// Matplotlib
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
// Python
#include <Python.h>


using data_type = float;
using feature_type = std::vector<data_type>;

constexpr int positive = 1;
constexpr int negative = -1;

inline bool all_zero(const feature_type& w, const data_type eps) {
    const int dimension = w.size();
    for(int i = 0;i < dimension; ++i)
        if(std::abs(w[i]) > eps)
            return false;
    return true;
}





class PerceptronClassifier {
private:
    feature_type weight;
    data_type bias;
    int dimension;
public:
    data_type forward(const feature_type& x) {
        data_type res = this->bias;
        for(int i = 0;i < dimension; ++i)
            res += this->weight[i] * x[i];
        return res;
    }

    std::vector<int> predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size();
        std::vector<int> prediction(samples_num);
        for(int i = 0;i < samples_num; ++i)
            prediction[i] = this->forward(input[i]) >= 0 ? positive: negative;
        return prediction;
    }

    void score(const std::vector<feature_type>& input, const std::vector<int>& label) {
        const auto prediction = this->predict(input);
        int correct = 0;
        const int samples_num = label.size();
        for(int i = 0;i < samples_num; ++i)
            correct += prediction[i] == label[i];
        printf("%d/%d====> [accuracy  %.3f]\n", correct, samples_num, correct * 1.0 / samples_num);
    }

    void backward(const feature_type& xi, const int yi, const data_type learning_rate) {
        for(int i = 0;i < dimension; ++i)
            this->weight[i] = this->weight[i] + yi * xi[i];
        this->bias = this->bias + learning_rate * yi;
    }

    void fit(const std::vector<feature_type>& input,
             const std::vector<int>& label,
             const int total_epochs=100,
             const data_type learning_rate=1e-3,
             const int seed=212) {
        // ??????????????????
        const int samples_num = label.size();
        assert(samples_num == input.size());
        this->dimension = input[0].size();
        // ?????????????????? bias
        if(this->weight.empty()) {
            this->weight.resize(dimension);
            std::default_random_engine e(seed);
            std::uniform_real_distribution<data_type> engine(-1, 1.0);
            for(int i = 0;i < dimension; ++i) this->weight[i] = engine(e);
            // ?????? weight ????????? 0
            while(all_zero(this->weight, 1e-7))
                for(int i = 0;i < dimension; ++i) this->weight[i] = engine(e);
            this->bias = 0;
        }
        // ????????????
        int epoch = 0;
        while(epoch++ <= total_epochs) {
            // ??????????????????
            int error_num = 0;
            for(int i = 0;i < samples_num; ++i) {
                const data_type output = this->forward(input[i]);
                if(label[i] * output < 0) {
                    this->backward(input[i], label[i], learning_rate);
                    ++error_num;
                }
            }
            std::cout << "error_num  " << error_num << std::endl;
            // ???????????????????????????, ??????
            if(error_num == 0) break;
        }
        for(int i = 0;i < dimension; ++i) std::cout << weight[i] << std::endl;
        std::cout << "bias = " << bias << std::endl;
        this->plot_classifier(input, label, total_epochs);
    }

    void plot_classifier(
            const std::vector<feature_type>& input,
            const std::vector<int> label,
            const int epoch) const {
        // ?????? Python ??????, ??????
        Py_SetPythonHome(L"D:/environments/Miniconda");
        // ???????????? W x + b = 0
        // ?????? x2 = -W1 / W2 * x1 - b / W2
        const data_type rate = - weight[0] / weight[1];
        const data_type b = - this->bias / weight[1];
        // x2 = rate * x1 + b
        // ???????????? x1 ?????????
        data_type min_value = input[0][0], max_value = input[0][0];
        const int samples_num = input.size();
        for(int i = 1;i < samples_num; ++i) {
            if(min_value < input[i][0]) min_value = input[i][0];
            if(max_value > input[i][0]) max_value = input[i][0];
        }
        feature_type x1(2), x2(2);
        x1[0] = min_value, x2[0] = rate * min_value + b;
        x1[1] = max_value, x2[1] = rate * max_value + b;
        plt::plot(x1, x2);
        // ??????
        feature_type X1, X2, Y1, Y2, XSV, YSV;
        for(int i = 0;i < samples_num; ++i) {
            // ??????????????????
            if(label[i] == 1) {
                X1.emplace_back(input[i][0]);
                Y1.emplace_back(input[i][1]);
            } else { // ?????????
                X2.emplace_back(input[i][0]);
                Y2.emplace_back(input[i][1]);
            }
        }
        plt::scatter(X1, Y1, 4, {{"c", "red"}});
        plt::scatter(X2, Y2, 4, {{"c", "green"}});
        plt::save("./output/epoch_" + std::to_string(epoch) + ".png", 600);
        plt::show();
        plt::clf();
        plt::cla();
        plt::close();
    }
};







class DualPerceptronClassifier {
private:
    data_type bias;
    feature_type __alpha;
    std::vector<int> __label;
    std::vector<feature_type> __input;
public:
    void fit(std::vector<feature_type>& input,
             std::vector<int>& label,
             const data_type learning_rate=1e-3) {
        // ????????????
        const int samples_num = label.size();
        assert(samples_num == input.size());
        const int dimension = input[0].size();
        feature_type alpha(samples_num, 0);
        this->bias = 0;
        // ?????????, ?????? Gram ??????, ????????????
        std::vector<feature_type> Gram(samples_num, feature_type(samples_num));
        for(int i = 0;i < samples_num; ++i) {
            for(int j = 0;j <= i; ++j) {
                data_type res = 0;
                for(int k = 0;k < dimension; ++k)
                    res += input[i][k] * input[j][k]; // ????????????????????????
                Gram[i][j] = Gram[j][i] = res;
            }
        }
        while(true) {
            int error_num = 0;
            for(int i = 0;i < samples_num; ++i) {
                // ?????? xi ????????? yi
                data_type sum_value = this->bias;
                for(int j = 0;j < samples_num; ++j)
                    sum_value += alpha[j] * label[j] * Gram[i][j];
                // ????????????????????????
                if(label[i] * sum_value <= 0) {  // ????????????????????????, ?????????????????? ?? ??? b ?????? 0, ????????? < 0 ??????????????????
                    alpha[i] = alpha[i] + learning_rate;
                    bias = bias + learning_rate * label[i];
                    ++error_num;
                }
            }
            std::cout << "error_num  " << error_num << std::endl;
            if(error_num == 0) break;
        }
        // ??????
        this->__alpha = alpha;
        this->__input = input;
        this->__label = label;
        if(dimension == 2) this->plot_classifier(input, label, 212);
    }

    std::vector<int> predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size();
        const int dimension = input[0].size();
        std::vector<int> prediction(samples_num);
        for(int i = 0;i < samples_num; ++i) {  // ????????????
            data_type res = this->bias;
            const int train_samples_num = this->__label.size();
            for(int j = 0;j < train_samples_num; ++j) { // ??????
                data_type temp = 0;
                for(int d = 0;d < dimension; ++d)  // ??????????????????????????????????????????
                    temp += input[i][d] * this->__input[j][d];
                res += this->__alpha[j] * this->__label[j] * temp; // ??i * yi * ??????
            }
            prediction[i] = res >= 0 ? positive: negative;
        }
        return prediction;
    }

    void score(const std::vector<feature_type>& input, const std::vector<int>& label) {
        const auto prediction = this->predict(input);
        int correct = 0;
        const int samples_num = label.size();
        for(int i = 0;i < samples_num; ++i)
            correct += prediction[i] == label[i];
        printf("%d/%d====> [accuracy  %.3f]\n", correct, samples_num, correct * 1.0 / samples_num);
    }

    void plot_classifier(
            const std::vector<feature_type>& input,
            const std::vector<int> label,
            const int epoch) const {
        // ?????? Python ??????, ??????
        Py_SetPythonHome(L"D:/environments/Miniconda");
        // ????????? weight
        const int samples_num = input.size();
        const int dimension = input[0].size();
        feature_type weight(dimension);
        for(int i = 0;i < samples_num; ++i)
            for(int d = 0;d < dimension; ++d)
                weight[d] += this->__alpha[i] * this->__label[i] * this->__input[i][d];
        for(int i = 0;i < dimension; ++i)
            std::cout << weight[i] << std::endl;
        std::cout << "bias = " << bias << std::endl;
        // ???????????? W x + b = 0
        // ?????? x2 = -W1 / W2 * x1 - b / W2
        const data_type rate = - weight[0] / weight[1];
        const data_type b = - this->bias / weight[1];
        // x2 = rate * x1 + b
        // ???????????? x1 ?????????
        data_type min_value = input[0][0], max_value = input[0][0];
        for(int i = 1;i < samples_num; ++i) {
            if(min_value < input[i][0]) min_value = input[i][0];
            if(max_value > input[i][0]) max_value = input[i][0];
        }
        feature_type x1(2), x2(2);
        x1[0] = min_value, x2[0] = rate * min_value + b;
        x1[1] = max_value, x2[1] = rate * max_value + b;
        plt::plot(x1, x2);

        // ??????
        feature_type X1, X2, Y1, Y2, XSV, YSV;
        for(int i = 0;i < samples_num; ++i) {
            // ??????????????????
            if(label[i] == 1) {
                X1.emplace_back(input[i][0]);
                Y1.emplace_back(input[i][1]);
            } else { // ?????????
                X2.emplace_back(input[i][0]);
                Y2.emplace_back(input[i][1]);
            }
        }
        plt::scatter(X1, Y1, 4, {{"c", "red"}});
        plt::scatter(X2, Y2, 4, {{"c", "green"}});
        plt::save("./output/epoch_" + std::to_string(epoch) + ".png", 600);
        plt::show();
        plt::clf();
        plt::cla();
        plt::close();
    }
};


/*
 * 1. ???????????????????????????????????????????????????, ????????????
 * 2. ????????????????????????????????? C ??????????????????, ????????????????????????
 */


int main() {
    std::setbuf(stdout, 0);

    // ????????????
    std::ifstream reader("./test/test.txt");
    // ??????
    int samples_num;
    reader >> samples_num;
    std::cout << "samples_num  " << samples_num << std::endl;
    float a, b; int c;
    std::vector<feature_type> X;
    X.reserve(samples_num);
    std::vector<int> Y(samples_num, 0);
    for(int i = 0;i < samples_num; ++i) {
        reader >> a >> b >> c;
        if(c != positive and c != negative)
            continue;
        feature_type one(2);
        one[0] = a;
        one[1] = b;
        X.emplace_back(one);
        Y[i] = c;
    }
    // ??????????????????
    reader.close();
    // ?????????????????????
    PerceptronClassifier classifier;
    classifier.fit(X, Y, 100, 1e-3, 200);
    classifier.score(X, Y);

    DualPerceptronClassifier dual_classifier;
    dual_classifier.fit(X, Y);
    dual_classifier.score(X, Y);
    return 0;
}