#include "FW-MAP.h"
#include <cassert>
#include <array>
#include <iostream>
#include <cmath>

struct sub_problem_test {
  std::array<double,2> cost; // simplex with two entries.
};

double max_fn_test(double* wi, FWMAP::YPtr _y, FWMAP::TermData term_data) // maximization oracle. Must copy argmax_y <a^{iy},[PAD(wi) kappa]> to y, and return the free term a^{iy}[d].
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  size_t* y = (size_t*) _y;

  if(wi[0] + sp->cost[0] < wi[1] + sp->cost[1]) {
    *y = 0;
  } else {
    *y = 1;
  }
  //std::cout << "compute solution on " << sp << " = " << *y << " with w = (" << wi[0] << "," << wi[1] << ")\n";
  return sp->cost[*y]; 
}

static void copy_fn_test(double* ai, FWMAP::YPtr _y, FWMAP::TermData term_data)
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  size_t* y = (size_t*) _y;
  assert(*y == 0 || *y == 1);
  ai[1-*y] = 0.0;
  ai[*y] = 1.0;
}

static double dot_product_fn_test(double* wi, FWMAP::YPtr _y, FWMAP::TermData term_data)
{
  sub_problem_test* sp = (sub_problem_test*) term_data;
  size_t* y = (size_t*) _y;
  assert(*y == 0 || *y == 1);
  return wi[*y];
}

int main()
{ 
  FWMAP s (2, 2, max_fn_test, copy_fn_test, dot_product_fn_test);//int d, int n, MaxFn max_fn, CopyFn copy_fn, DotProductFn dot_product_fn;

  sub_problem_test sp1;
  sp1.cost = {9.0,10.0};
  sub_problem_test sp2;
  sp2.cost = {20.0,0.0};

  s.SetTerm(0, &sp1, 2, nullptr, sizeof(size_t));
  s.SetTerm(1, &sp2, 2, nullptr, sizeof(size_t));

  double cost = s.Solve();
  
  double* lambda1 = s.GetLambda(0);
  double* lambda2 = s.GetLambda(1);

  std::cout << "\n\nsolution = ";
  for(int i=0; i<2; ++i) { std::cout << lambda1[i] << ","; }
  std::cout << "\n";
  for(int i=0; i<2; ++i) { std::cout << lambda2[i] << ","; }
  std::cout << "\n";

  assert(std::abs(cost - 10.0) < 10e-6);
  return 0; 
}

