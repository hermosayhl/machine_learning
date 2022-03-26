// C++
#include <list>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <iostream>
// self
#include "pipeline.hpp"

using namespace pipeline;

class NaiveBayersClassifier {
private:
    // 固有属性
    int K;    // 类别个数
    int dimension;  // 特征维度
    std::unordered_map<int, double> prior_probs;  // 先验概率
    std::unordered_map<int, std::vector< std::vector<double> > > cond_probs;  // 条件概率
    // 辅助用的
    std::unordered_map<int, double> prior_log_probs;  // 先验概率的 log 形式
public:
    void fit(const std::vector<feature_ptr>& input,
             const std::vector<int>& label,
             const data_type lambda=1.0) {
        // 获取样本信息
        const int samples_num = label.size();
        assert(samples_num == input.size());
        this->dimension = input[0]->length;
        // 获取类别数目
        std::cout << "获取类别数目" << std::endl;
        std::unordered_map<int, std::vector<int> > categories;
        for(int i = 0;i < samples_num; ++i)
            categories[label[i]].emplace_back(i); // 这样写慢了点
        this->K = categories.size();
        // 统计 P(Y = 0) 和 P(Y = 1)
        for(const auto& cate : categories)
            prior_probs[cate.first] = (cate.second.size() + lambda) * 1.0 / (samples_num + K * lambda);
        std::cout << "每一类的先验概率  :  \n";
        for(const auto& it : prior_probs) {
            prior_log_probs[it.first] = std::log(it.second);
            std::cout << it.first << "  " << it.second << std::endl;
        }
        using count_type = std::vector< std::vector<data_type> >;
        // 统计每一类数据, 每一维度数据在每个取值上的出现次数
        std::unordered_map<int, count_type> counter;
        counter.reserve(K);
        for(const auto& cate : prior_probs)  // 784 * 2
            counter.emplace(cate.first, count_type(dimension, std::vector<data_type>(2, 0)));
        // 开始统计
        cond_probs.reserve(K);
        for(const auto& cate : categories) {
            auto& cate_count = counter[cate.first];
            // 比如类别 2 的数据, 遍历 784 个维度,
            for(const auto pos : cate.second) {
                // 取出这个样本的数据
                const data_type* sample = input[pos]->data;
                for(int i = 0;i < dimension; ++i)
                    ++cate_count[i][sample[i]];
                // 第 i 个特征, 取值为 sample[i] 的次数 + 1, 这里是专为 mnist 设计的, 换个数据集就 GG
            }
            // 计算这种类别, 每个特征维度, 取值为 0 和 1 的概率
            cond_probs.emplace(cate.first, std::vector< std::vector<double> >(dimension, std::vector<double>(2, 0)));
            auto& cond = cond_probs[cate.first];
            for(int i = 0;i < dimension; ++i) {
                const int temp = cate.second.size() + 2 * lambda;
                cond[i][0] = std::log((cate_count[i][0] + lambda) * 1.0 / temp);
                cond[i][1] = std::log((cate_count[i][1] + lambda) * 1.0 / temp);
            } // std::cout << cond[400][0] << ", " << cond[400][1] << std::endl;
        }
        // 结束了, 每个类, 每个特征维度, 取值为 0, 1 的概率
        std::cout << "数据拟合结束 !\n";
    }

    int inference(const feature_ptr& sample) {
        // 取出样本数据
        data_type* x_ptr = sample->data;
        // 找概率最大对应的类别
        int max_class = -1;
        double max_value = -1e30;
        // 计算每一类下的概率
        for(const auto& cate : this->prior_probs) {
            double prob = this->prior_log_probs[cate.first]; // 概率相加, 加 log 了
            auto& cond = this->cond_probs[cate.first]; // 取出这一类, 在每个特征上, 为 0, 为 1 的概率
            for(int j = 0;j < this->dimension; ++j) {
                const data_type value = x_ptr[j];
                prob += cond[j][value]; // 对数概率相加, log 函数好慢啊
            }
            if(prob > max_value) {
                max_value = prob;
                max_class = cate.first;
            }
        }
        return max_class;
    }

    std::vector<int> predict(const std::vector<feature_ptr>& input) {
        const int samples_num = input.size(); // O(1)
        std::vector<int> prediction(samples_num, 0);
        for(int i = 0;i < samples_num; ++i)
            prediction[i] = this->inference(input[i]);
        return prediction;
    }

    void score(const std::vector<feature_ptr>& input, const std::vector<int>& label) {
        const auto prediction = this->predict(input);
        int correct = 0;
        const int samples_num = prediction.size();
        for(int i = 0;i < samples_num; ++i)
            correct += prediction[i] == label[i];
        printf("%d/%d====> [accuracy  %.3f]\n", correct, samples_num, correct * 1.0 / samples_num);
        // 这里可以写个混淆矩阵
        // 计算混淆矩阵
        std::vector<std::vector<int> > confusion_matrix(K, std::vector<int>(K, 0));
        for(int i = 0;i < samples_num; ++i)
            ++confusion_matrix[label[i]][prediction[i]];
        // 打印混淆矩阵
        std::cout << "混淆矩阵如下(行 label, 列 prediction) : \n";
        for(int i = 0;i < K; ++i) {
            std::cout << "\t";
            for(int j = 0;j < K; ++j)
                std::cout << confusion_matrix[i][j] << "\t";
            std::cout << std::endl;
        }
    }
};


/*
 * 1. 处理连续值 ? p(x_j = 0, 1 | Y) 符合某个均值为 u, 方差 σ 的高斯分布；第二种做法是把连续值按照区间划分成多个离散值, 然后分类, 有点麻烦
 * 2. 极大似然估计和贝叶斯估计 ? 拉普拉斯平滑, 其实是添加了拉普拉斯分布的贝叶斯估计, 加入了一个先验概率
 * 3. 我目前写的 bayers 是专为 mnist 设计写的, 如果写成可以处理所有离散数据的 bayers 就会慢得多, 以后复习的时候有时间再写吧
 * 4. 朴素贝叶斯的决策边界
 */


int main() {
    std::setbuf(stdout, 0);

    // 读取数据集
    auto dataset = read_datasets("mnist", true);
    const auto& train_data = dataset["train"];
    // 声明一个 naive_bayers
    NaiveBayersClassifier classifier;
    // 拟合数据
    classifier.fit(train_data.first, train_data.second);
    // 测试
    const auto& test_data = dataset["test"];
    classifier.score(test_data.first, test_data.second);
    return 0;
}