// C++
#include <random>
#include <vector>
#include <iostream>
// Matplotlib
#include "matplotlibcpp.h"

int main() {

    namespace plt = matplotlibcpp;
    plt::plot({1,3,2,4});
    plt::show();
    return 0;
}