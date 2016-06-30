#include "simpson_integral.h"
#include <iostream>

namespace {
    static auto constexpr N = 100000000;
}

int main()
{
    auto const func = myfunctional::make_functional([](double x) { return 1.0 / (2.0 * std::sqrt(x)); });
    simpsonmpi::SimpsonMpi<decltype(func)> s(func, N, 1.0, 4.0);
    s();
    
    return 0;
}