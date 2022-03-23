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
        // 获取样本信息
        const int samples_num = label.size();
        assert(samples_num == input.size());
        this->dimension = input[0].size();
        // 初始化权重和 bias
        if(this->weight.empty()) {
            this->weight.resize(dimension);
            std::default_random_engine e(seed);
            std::uniform_real_distribution<data_type> engine(-1, 1.0);
            for(int i = 0;i < dimension; ++i) this->weight[i] = engine(e);
            // 防止 weight 全部是 0
            while(all_zero(this->weight, 1e-7))
                for(int i = 0;i < dimension; ++i) this->weight[i] = engine(e);
            this->bias = 0;
        }
        // 开始训练
        int epoch = 0;
        while(epoch++ <= total_epochs) {
            // 遍历所有样本
            int error_num = 0;
            for(int i = 0;i < samples_num; ++i) {
                const data_type output = this->forward(input[i]);
                if(label[i] * output < 0) {
                    this->backward(input[i], label[i], learning_rate);
                    ++error_num;
                }
            }
            std::cout << "error_num  " << error_num << std::endl;
            // 如果没有误分类样本, 停止
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
        // 设置 Python 路径, 坑爹
        Py_SetPythonHome(L"D:/environments/Miniconda");
        // 绘制曲线 W x + b = 0
        // 曲线 x2 = -W1 / W2 * x1 - b / W2
        const data_type rate = - weight[0] / weight[1];
        const data_type b = - this->bias / weight[1];
        // x2 = rate * x1 + b
        // 找到输入 x1 的范围
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
        // 画点
        feature_type X1, X2, Y1, Y2, XSV, YSV;
        for(int i = 0;i < samples_num; ++i) {
            // 如果是正样本
            if(label[i] == 1) {
                X1.emplace_back(input[i][0]);
                Y1.emplace_back(input[i][1]);
            } else { // 负样本
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
        // 获取信息
        const int samples_num = label.size();
        assert(samples_num == input.size());
        const int dimension = input[0].size();
        feature_type alpha(samples_num, 0);
        this->bias = 0;
        // 初始化, 计算 Gram 矩阵, 缓存一下
        std::vector<feature_type> Gram(samples_num, feature_type(samples_num));
        for(int i = 0;i < samples_num; ++i) {
            for(int j = 0;j <= i; ++j) {
                data_type res = 0;
                for(int k = 0;k < dimension; ++k)
                    res += input[i][k] * input[j][k]; // 这里可以只算一遍
                Gram[i][j] = Gram[j][i] = res;
            }
        }
        while(true) {
            int error_num = 0;
            for(int i = 0;i < samples_num; ++i) {
                // 每个 xi 算一遍 yi
                data_type sum_value = this->bias;
                for(int j = 0;j < samples_num; ++j)
                    sum_value += alpha[j] * label[j] * Gram[i][j];
                // 如果存在错误样本
                if(label[i] * sum_value <= 0) {  // 这个等于号好坑啊, 因为一开始的 α 和 b 都是 0, 如果是 < 0 就直接退出了
                    alpha[i] = alpha[i] + learning_rate;
                    bias = bias + learning_rate * label[i];
                    ++error_num;
                }
            }
            std::cout << "error_num  " << error_num << std::endl;
            if(error_num == 0) break;
        }
        // 保存
        this->__alpha = alpha;
        this->__input = input;
        this->__label = label;
        if(dimension == 2) this->plot_classifier(input, label, 212);
    }

    std::vector<int> predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size();
        const int dimension = input[0].size();
        std::vector<int> prediction(samples_num);
        for(int i = 0;i < samples_num; ++i) {  // 每个样本
            data_type res = this->bias;
            const int train_samples_num = this->__label.size();
            for(int j = 0;j < train_samples_num; ++j) { // 遍历
                data_type temp = 0;
                for(int d = 0;d < dimension; ++d)  // 先算这个样本和历史样本的内积
                    temp += input[i][d] * this->__input[j][d];
                res += this->__alpha[j] * this->__label[j] * temp; // αi * yi * 内积
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
        // 设置 Python 路径, 坑爹
        Py_SetPythonHome(L"D:/environments/Miniconda");
        // 先获取 weight
        const int samples_num = input.size();
        const int dimension = input[0].size();
        feature_type weight(dimension);
        for(int i = 0;i < samples_num; ++i)
            for(int d = 0;d < dimension; ++d)
                weight[d] += this->__alpha[i] * this->__label[i] * this->__input[i][d];
        for(int i = 0;i < dimension; ++i)
            std::cout << weight[i] << std::endl;
        std::cout << "bias = " << bias << std::endl;
        // 绘制曲线 W x + b = 0
        // 曲线 x2 = -W1 / W2 * x1 - b / W2
        const data_type rate = - weight[0] / weight[1];
        const data_type b = - this->bias / weight[1];
        // x2 = rate * x1 + b
        // 找到输入 x1 的范围
        data_type min_value = input[0][0], max_value = input[0][0];
        for(int i = 1;i < samples_num; ++i) {
            if(min_value < input[i][0]) min_value = input[i][0];
            if(max_value > input[i][0]) max_value = input[i][0];
        }
        feature_type x1(2), x2(2);
        x1[0] = min_value, x2[0] = rate * min_value + b;
        x1[1] = max_value, x2[1] = rate * max_value + b;
        plt::plot(x1, x2);

        // 画点
        feature_type X1, X2, Y1, Y2, XSV, YSV;
        for(int i = 0;i < samples_num; ++i) {
            // 如果是正样本
            if(label[i] == 1) {
                X1.emplace_back(input[i][0]);
                Y1.emplace_back(input[i][1]);
            } else { // 负样本
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
 * 1. 可以写一下对偶感知机的核函数的例子, 真的可以
 * 2. 甚至我还可以写一个带有 C 惩罚的感知机, 允许近似线性可分
 */


int main() {
    std::setbuf(stdout, 0);

    // 读取数据
    std::ifstream reader("./test/test.txt");
    // 读写
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
    // 关闭文件资源
    reader.close();
    // 声明一个分类器
    PerceptronClassifier classifier;
    classifier.fit(X, Y, 100, 1e-3, 200);
    classifier.score(X, Y);

    DualPerceptronClassifier dual_classifier;
    dual_classifier.fit(X, Y);
    dual_classifier.score(X, Y);
    return 0;
}