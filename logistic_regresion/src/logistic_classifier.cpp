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


using data_type = double; // 如果要用 fast_exp 就一定要用 double, 不能用 float
using feature_type = std::vector<data_type>;

constexpr int positive = 1;
constexpr int negative = 0;

inline bool all_zero(const feature_type& w, const data_type eps) {
    const int dimension = w.size();
    for(int i = 0;i < dimension; ++i)
        if(std::abs(w[i]) > eps)
            return false;
    return true;
}


inline double fast_exp(const double y) {
    double d;
    *(reinterpret_cast<int*>(&d) + 0) = 0;
    *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
    return d;
}


class LogisticClassifier {
private:
    feature_type weight;
    data_type bias;
    int dimension;
public:

    std::vector< std::pair<int, data_type> > predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size();
        std::vector< std::pair<int, data_type> > prediction;
        prediction.reserve(samples_num);
        for(int i = 0;i < samples_num; ++i) {
            const auto output = this->forward(input[i]);
            const int judge = output >= 0 ? positive: negative;
            const data_type e_wxb = fast_exp(output);
            const data_type prob = e_wxb / (1 + e_wxb);  // 属于 1 类别的概率
            prediction.emplace_back(judge, prob);
        }
        return prediction;
    }

    void score(const std::vector<feature_type>& input, const std::vector<int>& label) {
        const auto prediction = this->predict(input);
        int correct = 0;
        const int samples_num = label.size();
        for(int i = 0;i < samples_num; ++i)
            correct += prediction[i].first == label[i];
        printf("%d/%d====> [accuracy  %.3f]\n", correct, samples_num, correct * 1.0 / samples_num);
    }

    data_type forward(const feature_type& x) {
        data_type res = this->bias;
        for(int i = 0;i < dimension; ++i)
            res += this->weight[i] * x[i];
        return res;
    }

    void fit(const std::vector<feature_type>& input,
             const std::vector<int>& label,
             const int total_epochs=100,
             const data_type learning_rate=1e-3,
             const std::pair<std::string, data_type> regularizer={"none", 0.0},
             const bool plot=false,
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
        std::vector<data_type> loss_history(total_epochs);
        int epoch = 0;
        while(epoch++ <= total_epochs) {
            // 遍历所有样本
            for(int i = 0;i < samples_num; ++i) {
                // 先计算 wx + b
                const data_type output = this->forward(input[i]);
                // 计算 e^{wx + b}
                const data_type e_wxb = fast_exp(output);
                // 更新
                const data_type temp = learning_rate * (label[i] - e_wxb / (1 + e_wxb));
                this->bias = this->bias + temp;
                for(int d = 0;d < dimension; ++d)
                    this->weight[d] = this->weight[d] + input[i][d] * temp;
            }
            // 是否使用正则化手段(位置是不是放在这里?)
            if(regularizer.first == "l2") {
                for(int d = 0;d < dimension; ++d)
                    this->weight[d] = this->weight[d] - learning_rate * regularizer.second * 2 * this->weight[d];
            }
            // 计算总损失
            data_type total_loss = 0;
            for(int i = 0;i < samples_num; ++i) {
                const data_type wxb = this->forward(input[i]);
                total_loss -= label[i] * wxb - std::log(1 + fast_exp(wxb));
            }
            loss_history[epoch - 1] = total_loss;
            std::cout << "epoch  " << epoch << "  loss  " << total_loss << std::endl;
        }
        for(int i = 0;i < dimension; ++i) std::cout << weight[i] << std::endl;
        std::cout << "bias = " << bias << std::endl;
        if(plot) {
            // 画损失函数
            this->plot_loss(loss_history);
            // 画决策平面
            if(this->dimension) this->plot_classifier(input, label, total_epochs);
        }
    }

    void plot_loss(const std::vector<data_type>& loss_history) {
        // 设置 Python 路径, 坑爹
        Py_SetPythonHome(L"D:/environments/Miniconda");
        const int epochs_num = loss_history.size();
        std::vector<data_type> X(epochs_num);
        for(int i = 0;i < epochs_num; ++i) X[i] = i + 1;
        plt::scatter(X, loss_history, 2, {{"c", "red"}});
        plt::save("./output/loss_history.png", 600);
        plt::show();
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






/* logsitic 中的负样本标签得是 0, 如果使用(二元交叉熵)最大似然的话
 *
 * 1. l1 正则化是什么鬼, 而且 l2 正则化也没法体现出来, 得找个例子
 * 2. logsitic 回归可以处理近似线性可分
 * 3. logsitic 可以做除了分类之外的事情吗?
 * 4. 牛顿法的 logsitic
 * 5. 为什么叫 logsitic 回归, 跟线性回归有啥关系
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
        if(c != negative and c != positive) c = 0;
        feature_type one(2);
        one[0] = a;
        one[1] = b;
        X.emplace_back(one);
        Y[i] = c;
    }
    // 关闭文件资源
    reader.close();
    // 声明一个分类器
    LogisticClassifier classifier;
    classifier.fit(X, Y, 200, 1e-2, {"l2", 1.0}, true, 212);
    classifier.score(X, Y);

    // 试试逻辑斯蒂回归可以得出类别的概率
    auto prediction = classifier.predict({{6.07, 0.33}, {2.88, -1.71}, {-0.74, -1.84}});
    for(const auto it : prediction)
        std::cout << it.first << "  prob:" << it.second << std::endl;

    return 0;
}