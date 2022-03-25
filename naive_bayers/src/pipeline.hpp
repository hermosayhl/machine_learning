// C++
#include <assert.h>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <filesystem>


namespace pipeline {
    // 首先定义向量
    using data_type = int;

    // 一个向量的定义
    class feature {
    public:
        // 储存的数据
        const int length;
        data_type *data;
    public:
        feature(const int _length): length(_length), data(new data_type[_length]) {}
        // 给定一个向量然后拷贝数据
        feature(const std::vector<data_type>& some_array) : length(some_array.size()) {
            this->data = new data_type[length];
            std::memcpy(this->data, some_array.data(), sizeof(data_type) * length);
        }
        // 析构函数
        ~feature() noexcept {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
            }
        }
        void print(const std::string& message="") const {
            std::cout << message << "==> ";
            for(int i = 0;i < length; ++i) std::cout << this->data[i] << " ";
            std::cout << std::endl;
        }
    };
    using feature_ptr = std::shared_ptr<feature>;


    void print(const feature_ptr& data, const int sign){
        for(int i = 0;i < 28; ++i) {
            for(int j = 0;j < 28; ++j)
                std::cout << data->data[i * 28 + j] << "  ";
            std::cout << std::endl;
        }
        std::cout << "分类标签  " << sign << std::endl;
    };

    // 制作和读取 mnist 数据集
    using dataset_type = std::pair< std::vector<feature_ptr>, std::vector<int> >;

    std::unordered_map<const char*, dataset_type> read_datasets(const std::string& dataset_name="mnist", const bool show=false) {
        // 如果是 mnist 数据集
        if(dataset_name == "mnist") {
            // cache
            std::filesystem::path mnist_cache = "../datasets/mnist/mnist.crane";
            // 声明数据集
            dataset_type train_data, test_data;
            // 如果之前没有制作过, 读取 txt 制作
            if(not std::filesystem::exists(mnist_cache)) {
                std::filesystem::path train_file("../datasets/mnist/train.txt");
                std::filesystem::path test_file("../datasets/mnist/test.txt");
                assert(std::filesystem::exists(train_file) and std::filesystem::exists(test_file));
                using uchar = unsigned char;
                // 读取训练文件和测试文件
                auto read_text = [](const std::filesystem::path& file_name)
                        -> dataset_type {
                    std::ifstream reader(file_name.c_str());
                    // 读取样本数目, 样本特征维度
                    int images_num, dimension;
                    reader >> images_num >> dimension;
                    std::cout << images_num << ", " << dimension << std::endl;
                    // 读取每一条样本
                    std::vector<feature_ptr> features;
                    std::vector<int> labels;
                    features.reserve(images_num);
                    labels.reserve(images_num);
                    for(int i = 0;i < images_num; ++i) {
                        int sign;
                        feature_ptr piece(new feature(dimension));
                        data_type* piece_ptr = piece->data;
                        for(int j = 0;j < dimension; ++j) {
                            reader >> sign;
                            piece_ptr[j] = sign;
                        }
                        features.emplace_back(piece);
                        reader >> sign;
                        labels.emplace_back(sign);
                    }
                    reader.close();
                    return std::make_pair(std::move(features), std::move(labels));
                };
                train_data = read_text(train_file);
                test_data = read_text(test_file);
                // 制作缓存
                std::ofstream cache_writer(mnist_cache, std::ios::binary | std::ios::out);
                // 首先写入长度
                const int train_size = train_data.second.size();
                const int dimension = train_data.first.front()->length;
                cache_writer.write(reinterpret_cast<const char *>(&train_size), sizeof(int));
                cache_writer.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
                // 写入 labels
                cache_writer.write(reinterpret_cast<const char *>(&train_data.second[0]), train_size * sizeof(int));
                for(int i = 0;i < train_size; ++i)
                    cache_writer.write(reinterpret_cast<const char *>(&train_data.first[i]->data[0]), dimension * sizeof(data_type));
                // 测试集同理
                const int test_size = test_data.second.size();
                cache_writer.write(reinterpret_cast<const char *>(&test_size), sizeof(int));
                cache_writer.write(reinterpret_cast<const char *>(&test_data.second[0]), test_size * sizeof(int));
                for(int i = 0;i < test_size; ++i)
                    cache_writer.write(reinterpret_cast<const char *>(&test_data.first[i]->data[0]), dimension * sizeof(data_type));
                // 关闭资源
                cache_writer.close();
                std::cout << "Mnist 数据集的缓存文件已被写入 " << mnist_cache << std::endl;
                std::cout << train_size << ", " << dimension << ", " << test_size << std::endl;
            }
            else {
                // 读取 cache
                std::ifstream cache_reader(mnist_cache, std::ios::binary);
                int train_size, test_size, dimension;
                cache_reader.read((char*)(&train_size), sizeof(int));
                cache_reader.read((char*)(&dimension), sizeof(int));
                // 读取训练数据
                std::vector<feature_ptr> train_features;
                train_features.reserve(train_size);
                for(int i = 0;i < train_size; ++i)
                    train_features.emplace_back(new feature(dimension));
                std::vector<int> train_labels(train_size, 0);
                cache_reader.read((char*)(&train_labels[0]), train_size * sizeof(int));
                for(int i = 0;i < train_size; ++i)
                    cache_reader.read((char*)(&train_features[i]->data[0]), dimension * sizeof(data_type));
                // 同理读取 test
                cache_reader.read((char*)(&test_size), sizeof(int));
                std::vector<feature_ptr> test_features;
                test_features.reserve(test_size);
                for(int i = 0;i < test_size; ++i)
                    test_features.emplace_back(new feature(dimension));
                std::vector<int> test_labels(test_size, 0);
                cache_reader.read((char*)(&test_labels[0]), test_size * sizeof(int));
                for(int i = 0;i < test_size; ++i)
                    cache_reader.read((char*)(&(test_features[i]->data[0])), dimension * sizeof(data_type));
                // 放到数据集中
                train_data.first = std::move(train_features);
                train_data.second = std::move(train_labels);
                test_data.first = std::move(test_features);
                test_data.second = std::move(test_labels);
                // 关闭资源
                cache_reader.close();
                std::cout << "成功从缓存 " << mnist_cache << " 中读取 Mnist 数据集!\n";
                std::cout << train_size << ", " << dimension << ", " << test_size << std::endl;
                // =======================================================================
                if(show) {
                    print(train_data.first[200], train_data.second[200]);
                    print(train_data.first[520], train_data.second[520]);
                }
            }
            // 这样的写法到底有没有调用析构 ?? 后面查一查
            return {{"train", std::move(train_data)}, {"test", std::move(test_data)}};
        }
        return {};
    }
}