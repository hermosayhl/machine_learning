// C++
#include <chrono>
#include <unordered_map>
// self
#include "kdtree.hpp"
#include "pipeline.hpp"


namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }
}


void test_kdtree() {
    // 首先定义若干向量
    std::vector<feature_ptr> train_features;
    train_features.emplace_back(new feature({2, 3, 7}));
    train_features.emplace_back(new feature({4, 3, 4}));
    train_features.emplace_back(new feature({6, 1, 4}));
    train_features.emplace_back(new feature({2, 4, 5}));
    train_features.emplace_back(new feature({0, 5, 7}));
    train_features.emplace_back(new feature({4, 0, 6}));
    train_features.emplace_back(new feature({7, 1, 6}));
    train_features.emplace_back(new feature({1, 4, 4}));
    train_features.emplace_back(new feature({2, 1, 3}));
    train_features.emplace_back(new feature({3, 1, 4}));
    train_features.emplace_back(new feature({5, 2, 5}));
    // 声明 KDTree
    KDTree tree;
    // 加载数据
    tree.load(train_features);
    // 开始测试最近邻
    feature_ptr target(new feature({1.8, 4, 3.5}));  // [2, 4, 2.5]
    // const auto nearest = tree.find_nearest(target);
    // std::get<1>(nearest)->print("检索结果 ");
    // 开始测试 K 近邻
    const auto k_nearest = tree.find_k_nearest(target, 6);
    for(const auto item : k_nearest) {
        std::get<1>(item)->print(std::to_string(std::get<2>(item)) + "===> ");
    }
}


/*
 * 正常的 KNN 还需要一些问题(这个得重新)
 * 1. 不同特征维度的数据归一化
 * 2. 权重根据距离做加权
 * 3. 数据不均衡(长尾分布)
 */
class KNNClassifier {
private:
    const int k;
    KDTree kd_tree;
    std::vector<int> labels;
public:
    KNNClassifier(const int _k) : k(_k) {}
    // 适应 train 数据
    void fit(pipeline::dataset_type& training_data) {
        // 构建 kdtree
        this->kd_tree.load(training_data.first);
        std::cout << "kdtree 构建完毕 !\n";
        this->labels.swap(training_data.second);
    }
    int predict(const feature_ptr& input) const {
        // 去 kdtree 中去找最接近的 k 个
        const auto k_nearest = this->kd_tree.find_k_nearest(input, this->k);
        // 从这 K 个去投票
        std::unordered_map<int, int> counter;
        // pipeline::print(testing_data[i], -1);
        for(const auto& item : k_nearest) {
            // pos, feature, distance, 我需要的是 pos, 从 labels 中去找
            const int pred = this->labels[std::get<0>(item)];
            ++counter[pred];
            // std::cout << pred << ", " << std::get<0>(item) << " " << std::to_string(std::get<2>(item)) << "\n";
        }
        // 找投票数目最多的
        int max_count = 0;
        int max_label = 0;
        for(const auto& item : counter) {
            if(max_count < item.second) {
                max_count = item.second;
                max_label = item.first;
            }
        }
        // std::cout << "最终的分类结果是  " << max_label << std::endl;
        return max_label;
    }
    // 给定一些测试数据找 K 近邻投票
    std::vector<int> predict(const std::vector<feature_ptr>& testing_data) const {
        // 遍历每个样本
        const int test_size = testing_data.size();
        std::vector<int> prediction(test_size, 0);
        for(int i = 0;i < test_size; ++i) {
            prediction[i] = this->predict(testing_data[i]);
            if((i + 1) % 100 == 0) std::cout << i + 1 << std::endl;
        }
        return prediction;
    }
    // 直接求评分
    float score(const pipeline::dataset_type& testing_data) {
        float accuracy = 0.00;
        const int testing_size = testing_data.second.size();
        for(int i = 0;i < testing_size; ++i) {
            const int pred = this->predict(testing_data.first[i]);
            accuracy += pred == testing_data.second[i];
            if((i + 1) % 100 == 0) {
                std::cout << i + 1 << "===>  " << std::setiosflags(std::ios::fixed) << std::setprecision(4) << "[accuracy : ";
                std::cout << accuracy * 1.f / (i + 1) << "]" << std::endl;
            }
        }
        return accuracy * 1.f / testing_size;
    }
    // 保存模型
    void save_model() const {}
    // 加载模型
    void load_model(const std::filesystem::path& model_path) {}
};







int main() {
    // 及时输出
    std::setbuf(stdout, 0);

    // 测试 kdtree
    // test_kdtree();

    // 准备数据集
    auto dataset = pipeline::read_datasets("mnist");

    // 准备 KNN
    KNNClassifier classifier(30);

    // 构建 Kdtree
    classifier.fit(dataset["train"]); // train 数据集报废了

    // 测试评分
    classifier.score(dataset["test"]);

    /* 在测试集合上测试
    std::vector<int> prediction;
    run([&](){
        prediction = classifier.predict(dataset["test"].first);
    }, "找 K 近邻 ");

    // 计算准确率
    float accuracy = 0.0;
    const auto& labels = dataset["test"].second;
    const int test_size = prediction.size();
    for(int i = 0;i < test_size; ++i) {
        accuracy += prediction[i] == labels[i];
    }
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4) << "测试集准确率  ";
    std::cout << accuracy * 1.f / test_size << std::endl;
    */

    // 保存模型
   	classifier.save_model();

    return 0;
}