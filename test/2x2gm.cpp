#include "../SVM.h"
#include <cassert>
#include <array>
#include <iostream>

using namespace BCFW_Bundle;

struct sub_problem_test {
  std::array<double,2> cost; // simplex with two entries.
  double sign;
};

double max_fn_test(double* wi, YPtr _y, TermData term_data) // maximization oracle. Must copy argmax_y <a^{iy},[PAD(wi) kappa]> to y, and return the free term a^{iy}[d].
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  const double sign = sp->sign;
  assert(sign == 1.0 || sign == -1.0);
  size_t* y = (size_t*) _y;

  if(sign*wi[0] + sp->cost[0] > sign*wi[1] + sp->cost[1]) {
    *y = 0;
  } else {
    *y = 1;
  }
  //std::cout << "compute solution on " << sp << " = " << *y << " with w = (" << wi[0] << "," << wi[1] << ")\n";
  return sp->cost[*y]; 
}

static bool compare_fn_test(YPtr _y1, YPtr _y2, TermData term_data)
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  size_t* y1 = (size_t*) _y1;
  assert(*y1 == 0 || *y1 == 1);
  size_t* y2 = (size_t*) _y2;
  assert(*y2 == 0 || *y2 == 1);
  return *y1 != *y2;
}

static void copy_fn_test(double* ai, YPtr _y, TermData term_data)
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  const double sign = sp->sign;
  assert(sign == 1.0 || sign == -1.0);
  size_t* y = (size_t*) _y;
  assert(*y == 0 || *y == 1);
  ai[1-*y] = 0.0;
  ai[*y] = sign*1.0;
}

static double dot_product_fn_test(double* wi, YPtr _y, TermData term_data)
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  const double sign = sp->sign;
  assert(sign == 1.0 || sign == -1.0);
  size_t* y = (size_t*) _y;
  assert(*y == 0 || *y == 1);
  return sign*wi[*y];
}

int main()
{ 
  SVM s (2, 2, max_fn_test, copy_fn_test, compare_fn_test, dot_product_fn_test, nullptr, true);//int d, int n, MaxFn max_fn, CopyFn copy_fn, CompareFn compare_fn, DotProductFn dot_product_fn, DotProductKernelFn dot_product_kernel_fn, bool zero_lower_bound);

  sub_problem_test sp1;
  sp1.sign = 1.0;
  sp1.cost = {9.0,10.0};
  sub_problem_test sp2;
  sp2.sign = -1.0;
  sp2.cost = {20.0,0.0};

  s.SetTerm(0, &sp1, 2, sizeof(size_t), nullptr );
  s.SetTerm(1, &sp2, 2, sizeof(size_t), nullptr );

  double* w = s.Solve();

  std::cout << "\n\nsolution = ";
  for(int i=0; i<2; ++i) {
    std::cout << w[i] << ",";
  }
  std::cout << "\n";

  return 0; 
}

