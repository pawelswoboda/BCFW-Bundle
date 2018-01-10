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


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "FW-MAP.h"
#include "utils.h"
#include "timer.h"


void FWMAP::Term::ComputeLambda(double* lambdai)
{
	int k;
	double* mui = fwmap->MU + shift;
	if (!mapping)
	{
		for (k=0; k<di; k++) lambdai[k] = fwmap->c*xi[k] + mui[k] - fwmap->nu[k];
	}
	else
	{
		for (k=0; k<di; k++) lambdai[k] = fwmap->c*xi[k] + mui[k] - fwmap->nu[mapping[k]];
	}
}

double FWMAP::GetCurrentUpperBound()
{
	double sum1 = 0, sum2 = 0, sum3 = 0;
	int i, k;
	for (i=0; i<n; i++)
	{
		double* xi = terms[i]->xi;
		double* mui = MU + terms[i]->shift;
		for (k=0; k<terms[i]->di; k++)
		{
			sum1 += xi[k]*xi[k];
			sum2 += xi[k]*mui[k];
		}
		sum2 += xi[k];
	}
	for (k=0; k<d; k++) sum3 += counts[k]*nu[k]*nu[k];
	return c*sum1/2 + sum2 - sum3/(2*c);
}

void FWMAP::ComputeGaps(double* LAMBDA, double& upper_bound, double& gap_factor, double* x_buf)
{
	double* x_min = x_buf;
	double* x_max = x_buf + d;

	int i, k;

	for (k=0; k<d; k++)
	{
		x_min[k] = 1e100;
		x_max[k] = -1e100;
	}

	upper_bound = 0;
	for (i=0; i<n; i++)
	{
		double* xi = terms[i]->xi;
		double* lambdai = LAMBDA + terms[i]->shift;
		int* mapping = terms[i]->mapping;
		for (k=0; k<terms[i]->di; k++)
		{
			int kk = (mapping) ? mapping[k] : k;
			if (x_min[kk] > xi[k]) x_min[kk] = xi[k];
			if (x_max[kk] < xi[k]) x_max[kk] = xi[k];
			upper_bound += xi[k]*lambdai[k];
		}
		upper_bound += xi[k];
	}

	gap_factor = 0;
	for (k=0; k<d; k++)
	{
		gap_factor += x_max[k] - x_min[k];
	}
}


void FWMAP::InitSolver()
{
	int i, k;

	int di_max = 0;
	int y_size_in_bytes_max = 0;
	di_sum = 0;
	for (i=0; i<n; i++)
	{
		if (di_max < terms[i]->di) di_max = terms[i]->di;
		if (y_size_in_bytes_max < terms[i]->y_size_in_bytes) y_size_in_bytes_max = terms[i]->y_size_in_bytes;
		terms[i]->shift = di_sum;
		di_sum += terms[i]->di;
	}

	LAMBDA_best = (double*) buf.Alloc(di_sum*sizeof(double));
	LAMBDA = (double*) buf.Alloc(di_sum*sizeof(double));
	MU = (double*) buf.Alloc(di_sum*sizeof(double));

	SetZero(nu, d);
	SetZero(LAMBDA_best, di_sum);
	SetZero(MU, di_sum);
	memset(counts, 0, d*sizeof(int));

	total_plane_num = 0;
	timestamp = 0;
	timestamp_threshold = -1;

	xi_buf = (double*) buf.Alloc((di_max+1)*sizeof(double));
	if (copy_fn) y_buf = (YPtr) buf.Alloc(y_size_in_bytes_max + sizeof(double));
	else         y_buf = (YPtr) xi_buf;

	v_best = 0;
	for (i=0; i<n; i++)
	{
		double* xi = terms[i]->xi;

		YPtr y = (copy_fn) ? y_buf : xi;
		xi[terms[i]->di] = *terms[i]->GetFreeTermPtr(y) = (*min_fn)(LAMBDA_best + terms[i]->shift, y, terms[i]->term_data);
		v_best += xi[terms[i]->di];
		if (copy_fn) (*copy_fn)(xi, y, terms[i]->term_data);
		if (!terms[i]->mapping)
		{
			for (k=0; k<d; k++)
			{
				nu[k] += c*xi[k];
				counts[k] ++;
			}
		}
		else
		{
			for (k=0; k<terms[i]->di; k++)
			{
				nu[terms[i]->mapping[k]] += c*xi[k];
				counts[terms[i]->mapping[k]] ++;
			}
		}
		AddCuttingPlane(i, y);
	}

	for (k=0; k<d; k++)
	{
		if (counts[k]) nu[k] /= counts[k];
	}
}


double FWMAP::Solve()
{
	double time_start = get_time();

	c = options.c;
	if (!LAMBDA_best)
	{
		InitSolver();
	}

	double* x_buf = new double[2*d];
	int* permutation = NULL;
	if (options.randomize_method >= 1 && options.randomize_method <= 3)	permutation = new int[n];
	if (options.randomize_method == 1) generate_permutation(permutation, n);

	int iter, approx_pass, total_pass;
	int _i, i, k;
	int approx_max = (options.cp_max <= 0) ? 0 : options.approx_max;

	double upper_bound_last = GetCurrentUpperBound();
	for (iter=total_pass=0; iter<options.iter_max; iter++)
	{
		timestamp = (float)(((int)timestamp) + 1); // When a plane is accessed, it is marked with 'timestamp'.
		                                           // Throughout the outer iteration, this counter will be gradually
		                                           // increased from 'iter+1' to 'iter+1.5', so that we
		                                           // (1) we can distinguish between planes added in the same iteration (when removing the oldest plane), and
		                                           // (2) we can easily determine whether a plane has been active during the last 'cp_inactive_iter_max' iterations
		if (options.cp_inactive_iter_max > 0) timestamp_threshold = timestamp - options.cp_inactive_iter_max;

		if (options.randomize_method == 2) generate_permutation(permutation, n);

		double _t[2];           // index 0: before calling real oracle
		double _upper_bound[2]; // index 1: after calling real oracle

		_t[0] = get_time();
		if (_t[0] - time_start > options.time_max) break;
		_upper_bound[0] = GetCurrentUpperBound();

		for (approx_pass=-1; approx_pass<approx_max; approx_pass++, total_pass++)
		{
			timestamp += (float) ( 0.5 / (approx_max+1) );

			if (options.randomize_method == 3) generate_permutation(permutation, n);

			for (_i=0; _i<n; _i++)
			{
				if (permutation)                        i = permutation[_i];
				else if (options.randomize_method == 0) i = _i;
				else                                    i = RandomInteger(n);
			
				double* xi = terms[i]->xi;
				double* mui = MU + terms[i]->shift;
				double* lambdai = LAMBDA;
				terms[i]->ComputeLambda(lambdai);
				int di = terms[i]->di;
				int* mapping = terms[i]->mapping;
				YPtr y_new = y_buf;
				double* xi_new = xi_buf;

				if (approx_pass < 0) // call real oracle
				{
					*terms[i]->GetFreeTermPtr(y_new) = (*min_fn)(lambdai, y_new, terms[i]->term_data);
					AddCuttingPlane(i, y_new);
				}
				else  // call approximate oracle
				{
					int t = terms[i]->Minimize(lambdai);
					terms[i]->UpdateStats(t);
					memcpy(y_new, terms[i]->y_arr[t], terms[i]->y_size_in_bytes + sizeof(double));
					terms[i]->RemoveUnusedPlanes();
				}
				if (copy_fn) { (*copy_fn)(xi_new, y_new, terms[i]->term_data); xi_new[di] = *terms[i]->GetFreeTermPtr(y_new); }

				// min_{gamma \in [0,1]} B*gamma*gamma - 2*A*gamma
				double A = xi[di] - xi_new[di], B = 0;
				for (k=0; k<di; k++)
				{
					double z = xi[k] - xi_new[k];
					A += lambdai[k]*z;
					B += z*z;
				}
				B *= c;
				double gamma;
				if (B<=0) gamma = (A <= 0) ? 0 : 1;
				else
				{
					gamma = A/B;
					if (gamma < 0) gamma = 0;
					if (gamma > 1) gamma = 1;
				}

				if (!mapping)
				{
					for (k=0; k<di; k++)
					{
						double old = xi[k];
						xi[k] = (1-gamma)*xi[k] + gamma*xi_new[k];
						nu[k] += c*(xi[k] - old) / counts[k];
					}
				}
				else
				{
					for (k=0; k<di; k++)
					{
						double old = xi[k];
						xi[k] = (1-gamma)*xi[k] + gamma*xi_new[k];
						nu[mapping[k]] += c*(xi[k] - old) / counts[mapping[k]];
					}
				}
				xi[di] = (1-gamma)*xi[di] + gamma*xi_new[di];
			}

			double t = get_time();
			upper_bound_last = GetCurrentUpperBound();
			//printf("upper_bound=%f\n", upper_bound_last);

			if (approx_pass >= 0)
			{
				if ( (_upper_bound[1] - upper_bound_last) * (_t[1]-_t[0]) * options.approx_limit_ratio
				   < (_upper_bound[0] - _upper_bound[1] ) * (t-_t[1])      ) { approx_pass ++; break; }
			}

			_t[1] = t;
			_upper_bound[1] = upper_bound_last;
		}

		if ((iter % options.MPBCFW_check_freq) == 0 && iter > 0)
		{
			double t = get_time();
			for (i=0; i<n; i++)
			{
				terms[i]->ComputeLambda(LAMBDA+terms[i]->shift);
			}
			double v = Evaluate(LAMBDA);
			printf("iter=%d t=%fs v=%f", iter, t - time_start, v);
			if (v_best < v)
			{
				memcpy(LAMBDA_best, LAMBDA, di_sum*sizeof(double));
				v_best = v;
				printf("!");
			}

			if ((iter % (options.MPBCFW_check_freq*options.MPBCFW_update_freq)) == 0)
			{
				double gap0, gap1, upper_bound;
				ComputeGaps(LAMBDA_best, upper_bound, gap1, x_buf);
				gap0 = upper_bound - v_best;
				printf(" Gaps: %f %f\nUpdating mu", gap0, gap1);

				// possibly, modify c here

				memset(nu, 0, d*sizeof(double));
				for (i=0; i<n; i++)
				{
					double* lambdai = LAMBDA_best + terms[i]->shift;
					double* mui = MU + terms[i]->shift;
					if (!terms[i]->mapping)
					{
						for (k=0; k<d; k++)
						{
							mui[k] = lambdai[k];
							nu[k] += lambdai[k] + c*terms[i]->xi[k];
						}
					}
					else
					{
						for (k=0; k<terms[i]->di; k++)
						{
							mui[k] = lambdai[k];
							nu[terms[i]->mapping[k]] += lambdai[k] + c*terms[i]->xi[k];
						}
					}
				}
				for (k=0; k<d; k++)
				{
					if (counts[k]) nu[k] /= counts[k];
				}
				if (gap0 <= options.gap0_max && gap1 <= options.gap1_max)
				{
					printf("\n");
					break;
				}
			}
			printf("\n");
		}
	}

	delete [] x_buf;
	if (permutation) delete [] permutation;
	return v_best;
}


