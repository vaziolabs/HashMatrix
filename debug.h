#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

void debugPrint(const std::string& msg) {
    std::cout << "[DEBUG] " << msg << std::endl;
}

#endif