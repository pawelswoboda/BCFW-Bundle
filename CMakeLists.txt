project(BCFW_Bundle)
cmake_minimum_required(VERSION 2.8.12)

# C++11 - not strictly needed, however there is a strange bug in clang when not using this flag (https://bugs.llvm.org/show_bug.cgi?id=13530#c3)
add_compile_options(-std=c++11)

add_library(SVM SVM.cpp Solve.cpp) 
add_subdirectory(test)
