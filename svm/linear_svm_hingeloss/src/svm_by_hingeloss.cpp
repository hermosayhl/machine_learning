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



class HingeLossSVM {
private:
    // 固有属性
    data_type bias = 0;
    feature_type weight;
    int dimension;
    // 缓冲区
    data_type output;
    data_type bias_grads = 0;
    feature_type weight_grads;
public:
    HingeLossSVM() {}

    data_type forward(const feature_type& input) const {
        data_type result = this->bias;
        for(int i = 0;i < this->dimension; ++i)
            result += this->weight[i] * input[i];
        return result;
    }

    void backward(const feature_type& xi, const data_type yi, const data_type C) {
        data_type* const w_ptr = this->weight.data();
        const data_type loss = 1 - yi * this->output;
        if(loss <= 0) {
            for(int i = 0;i < dimension; ++i)
                weight_grads[i] += w_ptr[i];
        } else {
            for(int i = 0;i < dimension; ++i) {
                weight_grads[i] += - C * yi * xi[i] + w_ptr[i];
                bias_grads += -C * yi;
            }
        }
    }

    void update_gradients(const data_type learning_rate) {
        for(int i = 0;i < dimension; ++i)
            weight[i] = weight[i] - learning_rate * weight_grads[i];
        bias = bias - learning_rate * bias_grads;
    }

    void clear_grads() {
        for(int i = 0;i < dimension; ++i) weight_grads[i] = 0;
        bias_grads = 0;
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

    void fit(const std::vector<feature_type>& input,
             const std::vector<int> label,
             const int total_epochs,
             const data_type C=1.0,
             const data_type learning_rate=1e-3,
             const int verbose=100) {
        // 获取样本数据集信息
        const int samples_num = input.size();
        this->dimension = input[0].size();
        // 给 weight 分配空间并初始化
        if(this->weight.empty()) {
            this->weight.resize(dimension);
            std::default_random_engine seed;
            seed.seed(212);
            std::normal_distribution<float> engine(0.0, 1.0);
            for(int o = 0;o < dimension; ++o) weight[o] = engine(seed);
            // 给梯度分配
            this->weight_grads.resize(dimension, 0);
        }
        // 直接训练
        int epoch = 0;
        int batch_size = 8;
        while(++epoch <= total_epochs) {
            // 遍历数据集
            for(int i = 0;i < samples_num; ++i) {
                // forward
                this->output = this->forward(input[i]);
                // backward
                this->backward(input[i], label[i], C);
                // 梯度更新
                if((i + 1) % batch_size == 0) {
                    this->update_gradients(learning_rate / batch_size);
                    this->clear_grads();
                }
            }
            data_type total_loss = 0;
            for(int i = 0;i < dimension; ++i)
                total_loss += 0.5 * weight[i] * weight[i];
            for(int i = 0;i < samples_num; ++i)
                total_loss += C * std::max(0.f, 1 - label[i] * this->forward(input[i]));
            // 这里可以打印一下总的 HingeLoss
            printf("%d/%d===>  [loss  %.4f]\n", epoch, total_epochs, total_loss);
            // 画图
            if(verbose > 0 and epoch % verbose == 0 and dimension == 2)
                this->plot_classifier(input, label, epoch);
        }
        for(int i = 0;i < dimension; ++i)
            std::cout << weight[i] << " ";
        std::cout << "\nbias = " << bias << std::endl;
        // 画图
        if(dimension == 2) this->plot_classifier(input, label, total_epochs);
    }

    std::vector<int> predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size();
        std::vector<int> prediction(samples_num, 0);
        for(int i = 0;i < samples_num; ++i)
            prediction[i] = this->forward(input[i]) >= 0 ? positive : negative;
        return prediction;
    }

    void score(const std::vector<feature_type>& input, const std::vector<int>& label) {
        const auto prediction = this->predict(input);
        const int samples_num = label.size();
        int correct = 0;
        for(int i = 0; i < samples_num; ++i)
            correct += prediction[i] == label[i];
        printf("%d/%d===> Accuracy : %.3f", correct, samples_num, correct * 1.f / samples_num);
    }
};



/*
 *
 *
 *  1. 还没尝试近似线性可分, 会不会震荡, 可以试试
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
    HingeLossSVM classifier;
    classifier.fit(X, Y, 400, 1.0, 1e-3, 0);
    classifier.score(X, Y);
    return 0;
}