// C++
#include <assert.h>
#include <list>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>



std::vector<const char*> str_split(const char* target, const char* pattern) {
    char * strc = new char[std::strlen(target) + 1];
    std::strcpy(strc, target);
    std::vector<const char*> res;
    char *token = std::strtok(strc, pattern);
    while (token != NULL) {
        res.emplace_back(token);
        token = std::strtok(NULL, pattern);
    }
    return res;
}

using data_type = float;
using feature_type = std::vector<data_type>;


inline data_type square(const data_type x) {
    return x * x;
}


template<typename T>
class NaiveBayersContinuous {
    using param_type = std::pair<data_type, data_type>; // 均值方差
    using cond_type = std::vector<param_type>; // 每一维度特征的均值方差
private:
    int K;
    int dimension;
    std::unordered_map<T, data_type> prior_probs;
    std::unordered_map<T, cond_type> cond_probs;
public:
    void fit(const std::vector<feature_type>& input,
             const std::vector<T>& label) {
        const int samples_num = label.size();
        assert(samples_num == input.size());
        this->dimension = input[0].size();
        // 需要学习 P(Y = 0) 和 P(Y = 1), 还有每一类, 在每一维特征上的条件概率分布, 假设为均值 u, 方差 sigma 的高斯分布
        std::unordered_map<T, std::list<int> > categories;
        for(int i = 0;i < samples_num; ++i)
            categories[label[i]].emplace_back(i); // 记住每一类的样本的位置
        // 从这里得到类别的先验概率
        this->K = categories.size();
        for(const auto& cate : categories) // 这里要记得 log
            this->prior_probs.emplace(cate.first, std::log(cate.second.size() * 1.0 / samples_num));
        // 开始统计
        this->cond_probs.reserve(this->K);
        for(const auto& cate : categories) {
            // 分配空间
            this->cond_probs.emplace(cate.first, cond_type(dimension));
            auto& cate_cond_probs = this->cond_probs[cate.first];
            // 拿到每一类别的样本数据
            for(int d = 0;d < dimension; ++d) {
                data_type mean = 0;
                data_type var = 0;
                // 计算这个类别数据在 d 维度上的均值
                for(const auto pos : cate.second)
                    mean += input[pos][d];
                mean /= cate.second.size(); // O(1)
                // 计算方差
                for(const auto pos : cate.second)
                    var += square(input[pos][d] - mean);
                var /= cate.second.size();  // O(1)
                // 保留根据高斯分布, 这个类别在维度在 d 上的均值和方差
                cate_cond_probs[d].first = mean;
                cate_cond_probs[d].second = var;
            }
        }
        std::cout << "拟合结束 !\n";
    }

    T inference(const feature_type& x) {
        // 取出这个样本的数据
        const data_type* const x_ptr = x.data();
        // 记录最大概率
        data_type max_prob = -1e30;
        T max_class;
        // 计算每一类的概率
        for(const auto& cate : this->prior_probs) {
            data_type prob = cate.second;
            auto& cate_cond_probs = this->cond_probs[cate.first];
            // 累加所有维度上的概率, 根据高斯分布得到当前值的概率
            for(int d = 0; d < dimension; ++d)
                prob = prob - std::log(std::sqrt(cate_cond_probs[d].second)) - square(x_ptr[d] - cate_cond_probs[d].first) / (2 * cate_cond_probs[d].second);
            if(prob > max_prob) {
                max_prob = prob;
                max_class = cate.first;
            }
        }
        return max_class;
    }

    std::vector<T> predict(const std::vector<feature_type>& input) {
        const int samples_num = input.size(); // O(1)
        std::vector<T> prediction(samples_num);
        for(int i = 0;i < samples_num; ++i)
            prediction[i] = this->inference(input[i]);
        return prediction;
    }

    void score(const std::vector<feature_type>& input, const std::vector<T>& label) {
        const auto prediction = this->predict(input);
        int correct = 0;
        const int samples_num = prediction.size();
        for(int i = 0;i < samples_num; ++i)
            correct += prediction[i] == label[i];
        printf("%d/%d====> [accuracy  %.3f]\n", correct, samples_num, correct * 1.0 / samples_num);
        // 这里可以写个混淆矩阵
        // 计算混淆矩阵
        std::unordered_map<T, std::unordered_map<T, int> > confusion_matrix;
        for(int i = 0;i < samples_num; ++i)
            ++confusion_matrix[label[i]][prediction[i]];
        // 打印混淆矩阵
        std::cout << "混淆矩阵如下(行 label, 列 prediction) : \n";
        for(const auto& i : this->prior_probs) {
            std::cout << "\t";
            for(const auto& j : this->prior_probs)
                std::cout << confusion_matrix[i.first][j.first] << "\t";
            std::cout << std::endl;
        }
    }
};








using dataset_type = std::pair< std::vector<feature_type>, std::vector<int> >;

std::unordered_map<std::string, dataset_type> read_pima_indians() {
    const char* file_name = "../datasets/pima-indians-diabetes.csv";
    std::ifstream reader(file_name);
    const int buffer_size = reader.seekg(0, std::ios::end).tellg();
    char *buffer = new char[buffer_size];
    reader.seekg(21, std::ios::beg).read(&buffer[0], static_cast<std::streamsize>(buffer_size));
    auto all_strings = str_split(buffer, "\n");
    delete buffer;
    buffer = nullptr;
    // 打乱数据
    std::shuffle(all_strings.begin(), all_strings.end(), std::default_random_engine(212));
    // const int samples_num = all_strings.size(); // 读数据不对劲
    const int samples_num = 768;
    std::cout << "samples_num  " << samples_num << std::endl;
    std::vector<feature_type> train_X, test_X;
    std::vector<int> train_Y, test_Y;
    const int train_size = int(samples_num * 0.8);  // 拆分数据
    train_X.reserve(train_size);
    train_Y.reserve(train_size);
    test_X.reserve(samples_num - train_size);
    test_Y.reserve(samples_num - train_size);
    // 解析数据, 从字符串到 float
    for(int i = 0;i < samples_num; ++i) {
        const auto res = str_split(all_strings[i], ",");
        const int feature_size = res.size();
        feature_type temp(feature_size - 1);
        for(int j = 0;j < feature_size - 1; ++j)
            temp[j] = std::atof(res[j]);
        if(i < train_size) {
            train_X.emplace_back(std::move(temp));
            train_Y.emplace_back(std::atoi(res[feature_size - 1]));
        } else {
            test_X.emplace_back(std::move(temp));
            test_Y.emplace_back(std::atoi(res[feature_size - 1]));
        }
    }
    return {{"train", dataset_type(train_X, train_Y)}, {"test", dataset_type(test_X, test_Y)}};
}





using char_type = std::pair< std::vector<feature_type>, std::vector<std::string> >;
std::unordered_map<const char*, char_type>
        read_iris() {
    const char* file_name = "../datasets/iris.data";
    std::ifstream reader(file_name);
    assert(reader.is_open());
    const int buffer_size = reader.seekg(0, std::ios::end).tellg();
    char *buffer = new char[buffer_size];
    reader.seekg(0, std::ios::beg).read(&buffer[0], static_cast<std::streamsize>(buffer_size));
    auto all_strings = str_split(buffer, "\n");
    delete buffer;
    buffer = nullptr;
    // 打乱数据
    std::shuffle(all_strings.begin(), all_strings.end(), std::default_random_engine(212));
    const int samples_num = all_strings.size(); // 读数据不对劲
    std::cout << "samples_num  " << samples_num << std::endl;
    std::vector<feature_type> train_X, test_X;
    std::vector<std::string> train_Y, test_Y;
    const int train_size = int(samples_num * 0.8);  // 拆分数据
    train_X.reserve(train_size);
    train_Y.reserve(train_size);
    test_X.reserve(samples_num - train_size);
    test_Y.reserve(samples_num - train_size);
    // 解析数据, 从字符串到 float
    for(int i = 0;i < samples_num; ++i) {
        const auto res = str_split(all_strings[i], ",");
        const int feature_size = res.size();
        feature_type temp(feature_size - 1);
        for(int j = 0;j < feature_size - 1; ++j)
            temp[j] = std::atof(res[j]);
        if(i < train_size) {
            train_X.emplace_back(std::move(temp));
            train_Y.emplace_back(res[feature_size - 1]);
        } else {
            test_X.emplace_back(std::move(temp));
            test_Y.emplace_back(res[feature_size - 1]);
        }
    }
    return {{"train", char_type(train_X, train_Y)}, {"test", char_type(test_X, test_Y)}};
}




/*
 * 连续型的朴素贝叶斯好处在于
 * 1. 程序好写, 类型比较好处理
 * 2. 不需要做平滑, 连续值直接根据高斯分布, 生成的概率都是大于 0 的
 * 坏处在于
 */



int main() {
    // 读取数据
    auto dataset = read_iris();
    // auto dataset = read_pima_indians(); // 对应下面改成 NaiveBayersContinuous<int>, 标签是 int
    const auto& train_data = dataset["train"];
    const auto& test_data = dataset["test"];

    // 声明一个贝叶斯分类器
    NaiveBayersContinuous<std::string> classifier;

    // 拟合数据
    classifier.fit(train_data.first, train_data.second);
    classifier.score(train_data.first, train_data.second);

    // 测试
    classifier.score(test_data.first, test_data.second);
    return 0;
}