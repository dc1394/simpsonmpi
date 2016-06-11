#include "simpson_integral.h"
#include <iostream>
#include <cstdint>

static auto constexpr N = 120000;

int main(int argc, char **argv)
{
    auto const func = myfunctional::make_functional([](double x) { return 1.0 / (2.0 * std::sqrt(x)); });
    simpsonmpi::SimpsonMpi<decltype(func)> s(func, N, 1.0, 4.0);
    s();
    
    return 0;
}