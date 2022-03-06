// C++
#include <list>
#include <cmath>
#include <memory>
#include <vector>
#include <cstring>
#include <assert.h>
#include <iostream>
#include <functional>




// 首先定义向量
using data_type = float;

// 空指针定义
constexpr int empty = -1;

// 一个向量的定义
class feature {
public:
    // 储存的数据
    const int length;
    data_type *data;
    // 节点的连接关系
    int dimension;
    int l_child = empty, r_child = empty;
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
    bool is_leaf() const { return l_child == empty and r_child == empty; }
};
using feature_ptr = std::shared_ptr<feature>;


inline data_type square(const data_type x) {
    return x * x;
}

// 计算 lhs, rhs 两个向量的欧氏距离
data_type compute_euclidean_distance(const feature_ptr& lhs, const feature_ptr& rhs) {
    const int length = lhs->length;
    data_type mse = 0;
    for(int i = 0;i < length; ++i) mse += square(lhs->data[i] - rhs->data[i]);
    return std::sqrt(mse);
}




namespace assist {
    template<typename T>
    class BoundedMaxHeap {
    private:
        const int k;
        int size = 0;
        std::vector<T> nodes;
    public:
        BoundedMaxHeap(const int _k): k(_k), nodes(_k) {}
        // 比较
        inline bool compare(const T& lhs, const T& rhs) {
            return lhs.first > rhs.first;
        }
        // 从上往下调整
        void shift_down(int cur) {
            while(cur * 2 + 1 < this->size) {
                int l_child = cur * 2 + 1;
                // 如果右孩子存在而且, 右孩子比左孩子还大
                if(l_child + 1 < this->size and this->compare(this->nodes[l_child + 1], this->nodes[l_child]))
                    ++l_child;
                // 现在开始往左右孩子中更大的挪动, 看是不是更小可以挪到下面去
                if(this->compare(this->nodes[l_child], this->nodes[cur])) {
                    std::swap(this->nodes[cur].first, this->nodes[l_child].first);
                    std::swap(this->nodes[cur].second, this->nodes[l_child].second);
                    cur = l_child;
                }
                else break; // 挪不动了, 到此为止
            }
        }
        // 插入新元素
        void emplace(const data_type value, const int index) {
            // 如果满了
            if(this->overflow()) {
                // 如果要加入的这个值比最差的还要大, 直接抛弃, 这里的比较没和上面的 compare 统一
                if(value >= this->worst())  return;
                else {  // 替换堆顶, 插入新值
                    // 交换堆顶和新元素
                    this->nodes[0].first = value;
                    this->nodes[0].second = index;
                    // 堆顶要从上往下调整
                    this->shift_down(0);
                }
                return;
            }
            // 还可以继续加节点
            int pos = this->size++;
            this->nodes[pos].first = value;
            this->nodes[pos].second = index;
            // 节点往上更新
            while(pos) {
                // 获取父亲节点
                const int parent = (pos - 1) / 2;
                // 如果当前节点比 parent 节点更大
                if(this->compare(this->nodes[pos], this->nodes[parent])) {
                    std::swap(this->nodes[pos].first, this->nodes[parent].first);
                    std::swap(this->nodes[pos].second, this->nodes[parent].second);
                    pos = parent;
                }
                else break;
            }
        }
        // 返回这个堆的顶的值
        data_type worst() const {
            assert(this->size);
            return this->nodes[0].first;
        }
        // 判断堆满了没
        inline bool overflow() const {
            return this->k == this->size;
        }
        // 转化成从小到大的一个列表
        std::list<T> transform() {
            std::list<T> result;
            // 逐个删除堆顶
            while(this->size) {
                // 记录堆顶
                result.emplace_front(this->nodes[0]);
                // 删除堆顶
                --this->size;
                std::swap(this->nodes[0].first, this->nodes[this->size].first);
                std::swap(this->nodes[0].second, this->nodes[this->size].second);
                // 交换之后, 堆顶从上往下调整
                this->shift_down(0);
            }
            return result;
        }
        // 打印
        void print(const std::string& message="堆排列如下 ") const {
            std::cout << message << std::endl;
            for(int i = 0;i < size; ++i)
                std::cout << this->nodes[i].first << ", " << this->nodes[i].second << std::endl;
        }
    };
}



class KDTree {
private:
    std::vector<feature_ptr> nodes; // 这个得固定, 不能变
    int nodes_num = 0;
    int feature_shape = 0;
    int root = empty;
    std::function<data_type(const feature_ptr&, const feature_ptr&)> distance_fun;
public:
    KDTree(const std::string& _metrics="euclidean") {
        if(_metrics == "euclidean")
            this->distance_fun = compute_euclidean_distance;
        else {
            std::cout << "暂时只能用欧氏距离; 其他形式的还没做出来! 如果要改, 拓展分支那个判断有待商榷\n";
            this->distance_fun = compute_euclidean_distance;
        }
    }
    void load(std::vector<feature_ptr>& _nodes) {
        // 保存数据
        this->nodes.swap(_nodes);
        this->nodes_num = this->nodes.size();
        this->feature_shape = this->nodes[0]->length;
        // std::cout << this->nodes_num << "  ==  " << this->feature_shape << std::endl;
        // 准备根节点覆盖了哪些点
        std::vector<int> scope(this->nodes_num, 0);
        for(int i = 0;i < this->nodes_num; ++i) scope[i] = i;
        // 递归构造 kdtree, 返回根节点位置
        this->root = this->build_tree(scope, 0);
    }
private:
    int build_tree(std::vector<int>& scope, const int depth) {
        // 如果分到这个区域的样本数目为 0, 返回
        if(scope.empty())
            return empty;
        // 当前要比较的第几维特征
        const int dimension = depth % this->feature_shape;
        // 对 scope 排序
        std::sort(scope.begin(), scope.end(), [this, dimension](const int lhs, const int rhs){
            return this->nodes[lhs]->data[dimension] < this->nodes[rhs]->data[dimension];
        });
        // for(const auto it : scope) this->nodes[it]->print();
        // 选出中位数
        const int scope_size = scope.size(); // C++11 之后这个操作是 O(1) 的
        int split = scope_size / 2;
        // 往前推进
        while(split > 0 and this->nodes[scope[split]]->data[dimension] == this->nodes[scope[split - 1]]->data[dimension])
            --split;
        // 拆分成两部分
        std::vector<int> left_scope(scope.begin(), scope.begin() + split);
        std::vector<int> right_scope(scope.begin() + split + 1, scope.end());
        // 记录这个节点比较的维度
        split = scope[split];
        this->nodes[split]->dimension = dimension;
        // this->nodes[split]->print("深度 " + std::to_string(depth));
        // 更新指针关系
        this->nodes[split]->l_child = this->build_tree(left_scope, depth + 1);
        this->nodes[split]->r_child = this->build_tree(right_scope, depth + 1);
        return split;
    }


    void blaze_a_way(int cur_node, const feature_ptr& target, std::list<int>& trace) const {
        while(true) {
            // 获取当前节点
            const auto& u = this->nodes[cur_node];
            // 经过的这个点加入路径
            trace.emplace_back(cur_node);
            // 获取当前节点比较的是第几维度
            const int dimension = u->dimension;
            // 看走哪边
            cur_node = target->data[dimension] < u->data[dimension] ? u->l_child: u->r_child;
            // 如果走到了空
            if(cur_node == empty) break;
        }
    }

public:
    using find_result = std::tuple<int, feature_ptr, data_type>;

    find_result find_nearest(const feature_ptr& target) const {
        target->print("检索 ");
        // 如果树是空的
        if(this->root == -1) {
            std::cout << "树是空的, 匹配不到任何数据 !\n";
            std::make_pair(new feature(1), 0.0);
        }
        // 准备一个栈, 记录经过的点
        std::list<int> trace;
        // 开辟一条搜索道路
        this->blaze_a_way(this->root, target, trace);
        // 看下最后一个经过的节点, 叶子节点的信息
        const int leaf = trace.back();
        trace.pop_back();
        // 当前距离 target 最近的节点下标
        int nearest = leaf;
        // 算下当前最短距离
        data_type min_distance = this->distance_fun(target, this->nodes[leaf]);
        this->nodes[leaf]->print("叶子节点的信息 ");
        std::cout << "当前跟叶子节点的距离  " << min_distance << std::endl;
        std::cout << "查看下栈的信息 \n";
        for(const auto it : trace) this->nodes[it]->print();
        // 开始沿着栈回溯
        while(not trace.empty()) {
            // 出栈一个节点
            const int top_pos = trace.back();
            const auto top = this->nodes[top_pos];
            trace.pop_back();
            top->print("出栈节点 ");
            // 查看这个节点是不是离 target 更近
            const data_type cur_distance = this->distance_fun(target, top);
            std::cout << "cur_distance  " << cur_distance <<   "  <====>  " << min_distance << "\n";
            if(cur_distance < min_distance) {
                min_distance = cur_distance;
                nearest = top_pos;
                std::cout << "这个节点和 target 的距离更小, 更新 min_distance " << min_distance << "\n";
            }
            // 检查这个节点的另一边没访问 有没有更近的
            if(not top->is_leaf()) {
                int dimension = top->dimension;
                const data_type attempt = std::abs(top->data[dimension] - target->data[dimension]);
                std::cout << "检查的维度是 " << dimension << " 看看到另一边的最短距离 " << attempt << " , 而目前最短距离  " << min_distance << std::endl;
                if(attempt <= min_distance) { // 这里的等号可不可取 ???? 其实不必取, 因为最短距离都这样了, 这里的等号可以去掉
                    std::cout << "因此, 另一边存在可能更近的点\n";
                    // 如果之前走的是左边, 现在走右边试试
                    int extend = target->data[dimension] < top->data[dimension] ? top->r_child : top->l_child;
                    // 如果这个拓展的节点不是空的
                    if(extend != empty) {
                        this->nodes[extend]->print("可拓展的这一边是 ");
                        // 查找这个拓展节点的树
                        this->blaze_a_way(extend, target, trace);
                        std::cout << "看看栈中多了什么元素  \n";
                        for(const auto it : trace) this->nodes[it]->print();
                    }
                }
            }
            else std::cout << "当前节点是叶子, 无法继续拓展 !\n";
        }
        std::cout << "最短距离是 " << min_distance << std::endl;
        return std::make_tuple(nearest, this->nodes[nearest], min_distance);
    }


    // 基本思路和上面的一致,
    // 但需要维护一个容量有限的大顶堆, 还需要判断 k > nodes_num , 然后上面在拓展那里, 需要判断是否满了 k, 要是没满, 继续扩展
    // 要是满了, 直接找 worst, 要是到另一个子分支的最短距离小于等于 worse(这里也不用等于), 就继续拓展那个分支
    std::list<find_result> find_k_nearest(const feature_ptr& target, const int k) const {
        // K 合不合理
        assert(k > 0 and k <= this->nodes_num and "选取的个数不合法!");
        // target->print("开始检索 ");
        // 设计一个 bounded 大顶堆
        assist::BoundedMaxHeap< std::pair<data_type, int> > min_nodes(k);
        // 准备一个栈, 记录经过的点
        std::list<int> trace;
        // 开辟一条搜索道路
        this->blaze_a_way(this->root, target, trace);
        // 看下最后一个经过的节点, 叶子节点的信息
        const int leaf = trace.back();
        trace.pop_back();
        // 当前距离 target 最近的节点下标
        feature_ptr nearest = this->nodes[leaf];
        // 记录这个点的距离, 放入最大堆做比较
        min_nodes.emplace(this->distance_fun(target, nearest), leaf);
        // 开始沿着栈回溯
        while(not trace.empty()) {
            // 出栈一个节点
            const int top_pos = trace.back();
            const auto top = this->nodes[top_pos];
            trace.pop_back();
            // 查看这个节点是不是离 target 更近
            const data_type cur_distance = this->distance_fun(target, top);
            // 记录当前点和 target 的距离, 看在不在前 k 个里面
            min_nodes.emplace(cur_distance, top_pos);
            // 检查这个节点的另一边没访问 有没有更近的
            if(not top->is_leaf()) {
                int dimension = top->dimension;
                const data_type attempt = std::abs(top->data[dimension] - target->data[dimension]);
                // 如果到另一边孩子的最短距离可以更新到前 k 项 or 或者堆还没满
                if(attempt < min_nodes.worst() or not min_nodes.overflow()) {
                    // 如果之前走的是左边, 现在走右边试试
                    int extend = target->data[dimension] < top->data[dimension] ? top->r_child : top->l_child;
                    // 如果这个拓展的节点不是空的
                    if(extend != empty) {
                        // 查找这个拓展节点的树
                        this->blaze_a_way(extend, target, trace);
                    }
                }
            }
        }
        std::list<find_result> result;
        // 转化成有序列表, 从小到大
        const auto sorted = min_nodes.transform();
        for(const auto & item : sorted)
            result.emplace_back(item.second, this->nodes[item.second], item.first);
        return result;
    }

};
