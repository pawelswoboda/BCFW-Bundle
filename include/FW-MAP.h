//   FWMAP, version 1.02

/*
    Copyright Vladimir Kolmogorov vnk@ist.ac.at 2014

    This file is part of FWMAP.

    FWMAP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    FWMAP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FWMAP.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef OAISJNHFOASFASFASFASFNVASF
#define OAISJNHFOASFASFASFASFNVASF

#include "block.h"

/*
Goal: minimize function
f(x) = \sum_{i=0}^{n-1} fi(x[A_i])
over some (possibly discrete) set x\in X.

A_i is some subset of {0,1,...,d-1}, and x[A_i] is the the restriction of x to the variables in this subset.
Function f depends on 'd' variables, term i depends on di=|A_i| variables where 1<=di<=d
It is assumed \min_xi [fi(xi)+<lambdai,xi>] can be solved efficiently for any given i and lambdai
*/

class FWMAP
{
public:
  typedef void *TermData; // pointer provided by the user in SetTerm()
  typedef void *YPtr; // pointer to an array of size 'y_size_in_bytes' that stores planes (i.e. labelings) for individual terms.


  typedef double (*MinFn)(double* lambdai, YPtr y, TermData term_data); // Min-oracle (to be implemented by the user).
  // Must copy argmin_{x} [f(x)+<lambdai,x>] to y (possibly in a compressed form), and return the free term f(x)

  // If compressed representations of planes are used, then the user must implement the following two functions

  typedef void (*CopyFn)(double* xi, YPtr y, TermData term_data); // copies the plane encoded by 'y' to vector xi of size di
  typedef double (*DotProductFn)(double* lambdai, YPtr y, TermData term_data); // returns <lambdai, xi> where xi is the vector encoded by 'y'


	// d = # of variables, n = # of terms.
	FWMAP(int d, int n, MinFn min_fn, CopyFn copy_fn=NULL, DotProductFn dot_product_fn=NULL);
	~FWMAP();

	// This function must be called for each i=0,1,...,n-1 (in an arbitrary order)
	//
	// term_data:       will be passed to all user-defined functions.
	// di:              the number of variables on which function fi depeneds
	// mapping:         array of size di, with mapping[k]\in {0,1,...,d-1}
	//                  k-th variable of function fi corresponds to mapping[k]-th variable of function f
	//                  If mapping[k] == NULL then di must equal d, and the code assumes that mapping[k]==k
	// y_size_in_bytes: the size of the array used to store solutions (labelings) for term fi
	//                  Using y_size_in_bytes=0 is equivalent to using y_size_in_bytes=di*sizeof(double)
	//                  Note, if y_size_in_bytes is different from these values then functions 'copy_fn' and 'dot_product_fn' must be implemented
	void SetTerm(int i, TermData term_data, int di, int* mapping=NULL, int y_size_in_bytes=0); 

  void init();

	//double Solve(); // returns the value of the objective function h(lambda)

  double do_descent_step(); // iterate until we update the new center point mu

	double* GetLambda(int i) { return LAMBDA_best + terms[i]->shift; } // returns pointer to array of size 'di'

	double Evaluate(double* LAMBDA); // returns the value of the objective function for given an internally stored solution. Expensive - calls n oracles!

	struct Options
	{
		Options() :
			/////////////////////////////
			// 1. TERMINATION CRITERIA //
			/////////////////////////////
			iter_max(100000),
			time_max(3600), // 1 hour
			gap0_max(1e-1),
			gap1_max(1e-3),
	
			//////////////////////////////
			// 2. PARAMETERS of MP-BCFW //
			//////////////////////////////
			randomize_method(2),

			approx_max(1000), // <--- probably will not be reached (due to the param below)
			approx_limit_ratio(1.0),

			cp_max(100), // <--- probably will not be reached (due to the param below). Can be decreased if memory is an issue
			cp_inactive_iter_max(10), // <--- PERHAPS THE MOST IMPORTANT PARAMETER:
			                          //      for how many iterations inactive planes are kept in memory

			//////////////////////////////////////////
			// 3. PARAMETERS OF THE PROXIMAL METHOD //
			//////////////////////////////////////////
			MPBCFW_check_freq(5),
			MPBCFW_update_freq(2),
			c(1.0)
		{
		};

		/////////////////////////////
		// 1. TERMINATION CRITERIA //
		/////////////////////////////

		int iter_max; // maximum total number of 'MP-BCFW' iterations (where one 'MP-BCFW' iteration consists of one exact pass and up to 'approx_max' approximate passes)
		double time_max; // maximum allowed time in seconds
		double gap0_max, gap1_max; // the code computes numbers gap0 and gap1 such that h(lambda_opt) - h(lambda_current) <= gap0 + gap1*||lambda_current - lambda_opt||_1.
		                           // terminate if gap0 <= gap0_max and gap1 <= gap1_max

		// the code computes values gap, g1, g2 such that
		//     f(w) - f_opt <= gap + g1*||w-w_opt||_1
		//     f(w) - f_opt <= gap + g2*||w-w_opt||_2
		// for any optimal solution w_opt, where w is the current solution and f(w) = \sum_{i=1}^n H_i(w).

		//////////////////////////////
		// 2. PARAMETERS of MP-BCFW //
		//////////////////////////////

		int randomize_method; // 0: use default order for every iteration (0,1,...,n-1)
		                      // 1: generate a random permutation, use it for every iteration
		                      // 2: generate a new random permutation at every iteration
		                      // 3: generate a new random permutation at every exact & approximate pass
		                      // 4: for every step sample example in {0,1,...,n-1} uniformly at random

		int approx_max; // >= 0. Each iter first performs one pass with calls to the 'real' oracle,
		                //       and then up to 'approx_max' passes with calls to the 'approximate' oracle
		                //       It is recommended to set it to a large number and rely on the criterion below.
		double approx_limit_ratio; // extra stopping criterion: approx. pass is stopped if
		                           //    approx_limit_ratio * (increase of the lower bound during B) / (time of B) 
		                           //                       < (increase of the lower bound during A) / (time of A)
		                           // where B corresponds to the last approx. pass and A corresponds to the sequence of steps
		                           // from the beginning of the current iter (including the exact pass) until B

		///////////////////////////////
		// cutting planes parameters //
		///////////////////////////////

		// If there are more than 'cp_max' planes then remove the plane that has been inactive the longest time.
		// (A plane is active when it is added or when it is returned by the approximate oracle.)
		// Also after each approximate oracle call remove a plane if it hasn't been active during the last 'cp_inactive_iter_max' outer iterations (including the current one)
		int cp_max; // >= 0. 
		int cp_inactive_iter_max;  // if == 0 then this option is not used (so 0 corresponds to +\infty)

		//////////////////////////////////////////
		// 3. PARAMETERS OF THE PROXIMAL METHOD //
		//////////////////////////////////////////
								   
		int MPBCFW_check_freq; // compute the cost h(LAMBDA) after every 'MPBCFW_check_freq' iterations of MP-BCFW (and update LAMBDA_best, if necessary)
		int MPBCFW_update_freq; // update center 'mu' after every 'MPBCFW_check_freq*MPBCFW_update_freq' iterations of MP-BCFW
		double c; // the weight of the proximal term is 1/(2c)
	} options;











//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

private:
	class Term
	{
	public:
	
		Term(int d, TermData term_data, int* mapping, int y_size_in_bytes, FWMAP* fwmap);
		~Term();

		int di;
		int* mapping;
		TermData term_data;
		int y_size_in_bytes;

		int shift; // = \sum_{j<i} d_j

		int num; // number of planes 'y'
		double* xi; // array of size di+1

		YPtr* y_arr; // y_arr[t] points to an array of size y_size_in_bytes + sizeof(double), 0<=t<num. Can be NULL (= not allocated)
		float* last_accessed;  // timestamps, of size num

		FWMAP* fwmap;

		///////////////////////////////////////////////////////////////////////////////////////

		bool isDuplicate(YPtr y);
		int AddPlane(YPtr y, int cp_max); // if num>=cp_max then the plane with the lowest 'counter' will be deleted
		                                     // and the new plane 'a' will be inserted instead.
		                                     // returns id of the added plane
		void DeletePlane(int t); // plane 'num-1' is moved to position 't'.

		int Minimize(double* lambdai); // returns id of the cutting plane 'x' that maximizes fi(x)+<lambdai,x>.

		void UpdateStats(int t); // increases 'counter' for 't' and decreases it for other planes, with parameter 'cp_history_size' (see implementation)

		void RemoveUnusedPlanes();

		double* GetFreeTermPtr(YPtr y) { return (double*) ( ((char*)y) + y_size_in_bytes); }

		void ComputeLambda(double* lambdai);

		////////////////////////////////////
	private:
		int num_max;
		char* my_buf;

		void Allocate(int num_max_new);
	};

	int d, n, di_sum;
	double c;
	double v_best;
	double* LAMBDA_best; // of size di_sum
	double* LAMBDA;      // of size di_sum
	double* MU;          // of size di_sum
	MinFn min_fn;
	CopyFn copy_fn;
	DotProductFn dot_product_fn;

	double* xi_buf; // of size 1 + max_i terms[i]->di. 
	YPtr y_buf; // of size max_i terms[i]->y_size_in_bytes_plus. Note, if copy_fn==NULL then pointers y_buf and xi_buf coincide
	Buffer buf;

	double* nu; // of size d
	int* counts; // of size d
	Term** terms; // of size n

  // iteration information
	double* x_buf;
  int* permutation;
	int iter, approx_pass, total_pass;
  int approx_max;
  double upper_bound_last;
  double time_start;


	float timestamp, timestamp_threshold;
	int total_plane_num;

	double GetCurrentUpperBound();
	void ComputeGaps(double* LAMBDA, double& upper_bound, double& gap_factor, double* x_buf); // x_buf must be of size 2*d
	void InitSolver();
	void AddCuttingPlane(int i, YPtr y);
};


#endif
