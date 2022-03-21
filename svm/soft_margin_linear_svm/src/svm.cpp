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


inline data_type square(const data_type x) {
    return x * x;
}

// xi, xj 做 Linear 核函数运算
inline data_type linear_kernel(const feature_type& lhs, const feature_type& rhs) {
    const int dimension = lhs.size();
    data_type sum_value = 0;
    for(int i = 0;i < dimension; ++i) sum_value += lhs[i] * rhs[i];
    return sum_value;
}


// 高斯核（径向基）核函数运算
inline data_type rbf_kernel(const feature_type& lhs, const feature_type& rhs) {
    const int dimension = lhs.size();
    // 先求分子二范数
    data_type l2 = 0;
    for(int i = 0;i < dimension; ++i)
        l2 += square(lhs[i] - rhs[i]);
    data_type sigma = 0.5;
    return std::exp(- l2 / (2 * sigma * sigma));
}

// α 是有上下界限的
inline data_type clip(data_type x, const data_type L, const data_type H) {
    if(x < L) x = L;
    else if(x > H) x = H;
    return x;
}



class SVMClassifier {
private:
    // W, b; 其中 W 可以通过 αi, yi, xi 求和得到
    std::vector<feature_type> support_input;
    feature_type support_alpha;
    std::vector<int> support_label;
    data_type bias;
    // kernel 函数
    std::function<data_type(const feature_type& lhs, const feature_type& rhs)> kernel_fun;
public:
    SVMClassifier() {}
    ~SVMClassifier() noexcept {}

    // 清空
    void clear() {
        this->bias = 0;
        std::vector<feature_type>().swap(this->support_input);
        std::vector<int>().swap(support_label);
        feature_type().swap(this->support_alpha);
    }

    // 预测结果
    std::vector<int> predict(const std::vector<feature_type>& input) const {
        // 测试样例个数
        const int samples_num = input.size();
        // 准备存放预测结果
        std::vector<int> prediction(samples_num, 0);
        // 支撑向量的个数
        const int support_samples_num = this->support_label.size();
        // 遍历每个样例
        for(int i = 0;i < samples_num; ++i) {
            // wx + b ==> \sum(αj * yj * xj * input[i])
            data_type g = this->bias;
            for(int j = 0;j < support_samples_num; ++j)
                g += this->support_alpha[j] * this->support_label[j] * this->kernel_fun(input[i], this->support_input[j]);
            // wx + b >= 0 的是正样本
            prediction[i] = g >= 0 ? positive : negative;
        }
        return prediction;
    }

    // 计算准确率
    void score(const std::vector<feature_type>& input, const std::vector<int>& label) const {
        int right = 0;
        const auto prediction = this->predict(input);
        const int samples_num = prediction.size(); // O(1)
        for(int j = 0;j < samples_num; ++j)
            right += prediction[j] == label[j];
        printf("%d/%d===> Accuracy  :  %.3f\n", right, samples_num, right * 1.f / samples_num);
    }


    // 拟合数据
    void fit(const std::vector<feature_type>& input,
             const std::vector<int>& label,
             const int max_iters=100,
             const std::string kernel_name = "linear",
             const data_type C=1.0,
             const data_type eps=1e-5,
             const bool plot=false,
             const bool use_buffer=false,
             const bool verbose=false) {
        // 清空上次的参数
        this->clear();
        // 根据参数决定核函数
        if(kernel_name == "linear")
            this->kernel_fun = linear_kernel;
        else if(kernel_name == "rbf")
            this->kernel_fun = rbf_kernel;
        else assert(false and "暂未实现");
        // 获取样本数据集信息
        const int samples_num = input.size();
        std::cout << samples_num << " === " << label.size() << std::endl;
        assert(samples_num == label.size());
        const int dimension = input[0].size();
        // 初始化一些参数
        feature_type alpha = feature_type(samples_num, 0);
        // 定义函数 g(x) = sum(ai * yi * kernel(x[i], x[j])) + bias
        auto g = [&](const feature_type& x) ->data_type {
            data_type res = this->bias;
            for(int k = 0;k < samples_num; ++k)
                res += alpha[k] * label[k] * this->kernel_fun(x, input[k]);
            return res;
        };
        // 如果需要使用缓冲区, 这个占据的空间比较大, 如果样本数很大
        std::vector<feature_type> K;
        if(use_buffer) {
            K.reserve(samples_num);
            for(int i = 0;i < samples_num; ++i)
                K.emplace_back(feature_type(samples_num, 0));
            for(int i = 0;i < samples_num; ++i)
                for(int j = 0;j < samples_num; ++j)
                    K[i][j] = this->kernel_fun(input[i], input[j]);
        }
        // 选定一个 α1, 准备选 α2, 该咋选
        std::default_random_engine e(26);
        std::uniform_int_distribution<int> engine(0, samples_num - 1);
        // 准备一个 E 列表
        feature_type E_list(samples_num);
        // 开始迭代一定次数求 SMO
        int cur_iter = -1;
        while(++cur_iter <= max_iters) {
            // 直接遍历数据集, 找违反 KKT 条件的
            for(int i = 0;i < samples_num; ++i) {
                // 如果没有违反 KKT 条件
                const data_type gxi = g(input[i]);
                if((std::abs(alpha[i]) < eps and gxi * label[i] >= 1) or
                   (std::abs(alpha[i] - C) < eps and gxi * label[i] <= 1) or
                   (0 < alpha[i] and alpha[i] < C and std::abs(gxi * label[i] - 1) < eps))
                    continue;
                // 目前 α1 是违反 KKT 条件的
                // 找 α2
                /* 这种方式行不通
                int j;
                E_list[i] = gxi - label[i];
                if(E_list[i] >= eps) {
                    // 如果 E1 是正数, 找最小值
                    int min_index = 0;
                    data_type min_value = E_list[min_index];
                    for(int k = 1;k < samples_num; ++k)
                        if(E_list[k] < min_value and k != i)
                            min_value = E_list[k], min_index = k;
                    j = min_index;
                } else {
                    // 如果 E1 是负数, 找最大值
                    int max_index = 0;
                    data_type max_value = E_list[max_index];
                    for(int k = 1;k < samples_num; ++k)
                        if(E_list[k] > max_value and k != i)
                            max_value = E_list[k], max_index = k;
                    j = max_index;
                }
                const data_type Ei = E_list[i];
                const data_type Ej = E_list[j] = g(input[j]) - label[j];
                */
                // (这个随机是简单的)
                int j = engine(e);
                while(j == i) j = engine(e);
                // 取出这两个 α
                data_type alpha_1 = alpha[i];
                data_type alpha_2 = alpha[j];
                // 计算上下限
                data_type L, H;
                if(label[i] == label[j])
                    L = std::max(0.f, alpha_1 + alpha_2 - C),
                    H = std::min(C, alpha_1 + alpha_2);
                else
                    L = std::max(0.f, alpha_2 - alpha_1),
                    H = std::min(C, C + alpha_2 - alpha_1);
                // 如果区间没什么空间了, 不做后面的计算
                if(L > H or std::abs(L - H) < eps) continue;
                // 计算一些中间变量
                data_type Kii, Kjj, Kij;
                if(use_buffer)
                    Kii = K[i][i], Kjj = K[j][j], Kij = K[i][j];
                else {
                    Kii = this->kernel_fun(input[i], input[i]);
                    Kjj = this->kernel_fun(input[j], input[j]);
                    Kij = this->kernel_fun(input[i], input[j]);
                }
                // 计算 alpha 的分母
                const data_type eta = Kii + Kjj - 2 * Kij;
                // 计算 alpha_2
                const data_type Ei = gxi - label[i];
                const data_type Ej = g(input[j]) - label[j];
                data_type alpha_2_new = clip(alpha_2 + label[j] * (Ei - Ej) / eta, L, H);
                // 如果这次 α2 没啥变化, 那其它的也不会有啥变化
                if(std::abs(alpha_2_new - alpha_2) < eps) continue;
                // 计算 alpha_1
                data_type alpha_1_new = alpha_1 + label[i] * label[j] * (alpha_2 - alpha_2_new);
                // 更新 α1 和 α2
                alpha[i] = alpha_1_new;
                alpha[j] = alpha_2_new;
                // 计算 bias
                data_type b1 = this->bias - Ei - label[i] * (alpha_1_new - alpha_1) * Kii
                        - label[j] * (alpha_2_new - alpha_2) * Kij;
                data_type b2 = this->bias - Ei - label[i] * (alpha_1_new - alpha_1) * Kij
                        - label[j] * (alpha_2_new - alpha_2) * Kjj;
                // 具体见《统计学习方法》P148
                if(0 < alpha_1_new and alpha_1_new < C) this->bias = b1;
                else if(0 < alpha_2_new and alpha_2_new < C) this->bias = b2;
                else this->bias = (b1 + b2) / 2;
                /* 在这里更新 Ei, Ej
                E_list[i] = g(input[i]) - label[i];
                E_list[j] = g(input[j]) - label[j];
                */
            }
            // 如果有缓冲区, 计算损失
            if(use_buffer and verbose) {
                data_type loss_value = 0;
                for(int i = 0;i < samples_num; ++i) {
                    const data_type temp = alpha[i] * label[i];
                    for(int j = 0;j < samples_num; ++j)
                        loss_value += temp * alpha[j] * label[j] * K[i][j];
                    loss_value -= alpha[i];
                }
                if(cur_iter % 100 == 0) printf("[iter %d/%d]===>  [loss %.3f]\n", cur_iter + 1, max_iters, loss_value);
            }
        }
        // 拟合结束, 记录支撑向量
        std::vector<int> book(samples_num, 0);
        for(int i = 0;i < samples_num; ++i)
            if(alpha[i] > eps) book[i] = 1;
        this->support_input.reserve(samples_num);
        this->support_label.reserve(samples_num);
        this->support_alpha.reserve(samples_num);
        for(int i = 0;i < samples_num; ++i) {
            if(book[i]) {
                this->support_input.emplace_back(input[i]);
                this->support_alpha.emplace_back(alpha[i]);
                this->support_label.emplace_back(label[i]);
                std::cout << "alpha = " << alpha[i] << std::endl;
            }
        }
        std::cout << "数据拟合结束 !\n";
        std::cout << "支持向量  " << this->support_label.size() << "  个\n";
        for(const auto& it : this->support_input) {
            for(int i = 0;i < dimension; ++i)
                std::cout << it[i] << "  ";
            std::cout << std::endl;
        }
        if(plot == true and kernel_name == "linear" and dimension == 2) {
            // 先得到 W =  sum(αi, yi, xi)
            feature_type W(dimension, 0);
            const int support_num = this->support_input.size();
            for(int d = 0;d < dimension; ++d) {
                for(int i = 0;i < support_num; ++i) {
                    W[d] += this->support_input[i][d] * this->support_alpha[i] * this->support_label[i];
                }
            }
            for(int d = 0;d < dimension; ++d)
                std::cout << W[d] << std::endl;
            // 曲线 W x + b = 0
            // 曲线 x2 = -W1 / W2 * x1 - b / W2
            const data_type rate = - W[0] / W[1];
            const data_type b = - this->bias / W[1];
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
            // 设置 Python 路径, 坑爹
            Py_SetPythonHome(L"D:/environments/Miniconda");
            // 绘制曲线
            plt::plot(x1, x2);
            // 画点
            feature_type X1, X2, Y1, Y2, XSV, YSV;
            for(int i = 0;i < samples_num; ++i) {
                // 如果是支撑向量
                if(book[i] == 1) {
                    XSV.emplace_back(input[i][0]);
                    YSV.emplace_back(input[i][1]);
                } else {
                    // 如果是正样本
                    if(label[i] == 1) {
                        X1.emplace_back(input[i][0]);
                        Y1.emplace_back(input[i][1]);
                    } else { // 负样本
                        X2.emplace_back(input[i][0]);
                        Y2.emplace_back(input[i][1]);
                    }
                }
            }
            plt::scatter(X1, Y1, 4, {{"c", "black"}});
            plt::scatter(X2, Y2, 4, {{"c", "green"}});
            plt::scatter(XSV, YSV, 4, {{"c", "red"}});
            plt::save("./output/kernel_ + " + kernel_name + "_iters_" + std::to_string(max_iters) + ".png", 600);
            plt::show();
        }
    }
};




/*
 * 待做的东西
 * 1. Mnist 的分类, 这就成了多类分类了
 * 2. 模型保存以及模型加载
 * 3. LinearBaseSVM 这个的限制条件
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
    SVMClassifier classifier;
    // 拟合数据集
    classifier.fit(X, Y,
        1000,
        "linear",
        1.0,
        1e-5,
        true,
        true,
        true);
    // 预测
    classifier.score(X, Y);
    return 0;
}