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



//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

FWMAP::FWMAP(int _d, int _n, MinFn _min_fn, CopyFn _copy_fn, DotProductFn _dot_product_fn) :
	d(_d), n(_n), di_sum(0),
	LAMBDA_best(NULL), LAMBDA(NULL),
	min_fn(_min_fn), copy_fn(_copy_fn), dot_product_fn(_dot_product_fn), 
	buf(1024)
{
	int i;
	nu = (double*) buf.Alloc(d*sizeof(double));
	counts = (int*) buf.Alloc(d*sizeof(int));
	terms = new Term*[n];
	for (i=0; i<n; i++) terms[i] = NULL;
}

FWMAP::~FWMAP()
{
	if (terms)
	{
		int i;
		for (i=0; i<n; i++)
		{
			if (terms[i]) delete terms[i];
		}
		delete [] terms;
	}

	delete [] x_buf;
  if(permutation != nullptr) {
    delete [] permutation;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

double FWMAP::Evaluate(double* LAMBDA)
{
	int i;

	double v = 0;
	double* lambdai = LAMBDA;

	for (i=0; i<n; lambdai+=terms[i]->di, i++)
	{
		v += (*min_fn)(lambdai, y_buf, terms[i]->term_data);
		if (dot_product_fn)
		{
			v += (*dot_product_fn)(lambdai, y_buf, terms[i]->term_data);
		}
		else
		{
			v += DotProduct(lambdai, (double*)y_buf, terms[i]->di);
		}
	}

	return v;
}





void FWMAP::SetTerm(int i, TermData term_data, int di, int* mapping, int y_size_in_bytes)
{
	if (terms[i]) { printf("Error: SetTerm() cannot be called twice for the same term\n"); exit(1); }
	if (!mapping && di != d) { printf("Error in SetTerm(): di must equal d when mapping==NULL\n"); exit(1); }
	if (di<1 || di>d)  { printf("Error in SetTerm(): di out of bounds\n"); exit(1); }

	if (y_size_in_bytes <= 0) y_size_in_bytes = di*sizeof(double);

	terms[i] = new Term(di, term_data, mapping, y_size_in_bytes, this);
}



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
////////////////////// Implementation of 'Term' /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////





FWMAP::Term::Term(int _d, TermData _term_data, int* _mapping, int _y_size_in_bytes, FWMAP* _svm)
	: di(_d), term_data(_term_data), mapping(_mapping), y_size_in_bytes(_y_size_in_bytes),
	  fwmap(_svm), num(0), num_max(0)
{
	xi = (double*) fwmap->buf.Alloc((di+1)*sizeof(double));
	y_arr = NULL;
	last_accessed = NULL;
	my_buf = NULL;
	Allocate(4); // start with up to 4 planes per term, then allocate more if necessary
}

void FWMAP::Term::Allocate(int num_max_new)
{
	int num_max_old = num_max;
	YPtr* y_arr_old = y_arr;
	float* last_accessed_old = last_accessed;
	char* my_buf_old = my_buf;
	num_max = num_max_new;

	int i, my_buf_size = num_max*sizeof(YPtr) + num_max*sizeof(float);
	my_buf = new char[my_buf_size];

	y_arr = (YPtr*) my_buf;
	for (i=0; i<num_max_old; i++) y_arr[i] = y_arr_old[i];
	for ( ; i<num_max; i++) y_arr[i] = NULL;

	last_accessed = (float*) (y_arr + num_max);
	memcpy(last_accessed, last_accessed_old, num_max_old*sizeof(int));

	if (my_buf_old) delete [] my_buf_old;
}


FWMAP::Term::~Term()
{
	if (my_buf) delete [] my_buf;
}

bool FWMAP::Term::isDuplicate(YPtr y)
{
	int t;

	for (t=0; t<num; t++)
	{
		if (!memcmp(y, y_arr[t], y_size_in_bytes))
		{
			last_accessed[t] = fwmap->timestamp;
			return true;
		}
	}
	return false;
}

int FWMAP::Term::AddPlane(YPtr y, int cp_max)
{
	int t, t2;

	if (num >= cp_max)
	{
		for (t=0, t2=1; t2<num; t2++)
		{
			if (last_accessed[t] > last_accessed[t2]) t = t2;
		}
	}
	else
	{
		if (num >= num_max)
		{
			int num_max_new = 2*num_max+1; if (num_max_new > cp_max) num_max_new = cp_max;
			Allocate(num_max_new);
		}
		t = num ++;
		if (y_arr[t] == NULL) y_arr[t] = (YPtr) fwmap->buf.Alloc(y_size_in_bytes + sizeof(double));
		fwmap->total_plane_num ++;
	}
	memcpy(y_arr[t], y, y_size_in_bytes + sizeof(double));
	last_accessed[t] = fwmap->timestamp;

	return t;
}

void FWMAP::Term::DeletePlane(int t)
{
	num --;
	fwmap->total_plane_num --;
	if (t == num) return;
	YPtr tmp = y_arr[t]; y_arr[t] = y_arr[num]; y_arr[num] = tmp;
	last_accessed[t] = last_accessed[num];
}

int FWMAP::Term::Minimize(double* lambdai)
{
	int t_best, t;
	double v_best;
	if (fwmap->dot_product_fn)
	{
		for (t=0; t<num; t++)
		{
			double v = (*fwmap->dot_product_fn)(lambdai, y_arr[t], term_data) + (*GetFreeTermPtr(y_arr[t]));
			if (t == 0 || v_best >= v) { v_best = v; t_best = t; }
		}
	}
	else
	{
		for (t=0; t<num; t++)
		{
			double v = DotProduct(lambdai, (double*)y_arr[t], di) + (*GetFreeTermPtr(y_arr[t]));
			if (t == 0 || v_best >= v) { v_best = v; t_best = t; }
		}
	}
	return t_best;
}

void FWMAP::Term::UpdateStats(int t_best)
{
	last_accessed[t_best] = fwmap->timestamp;
}

void FWMAP::AddCuttingPlane(int i, YPtr y)
{
	if (options.cp_max <= 0) return;
	if (terms[i]->isDuplicate(y)) return;
	terms[i]->AddPlane(y, options.cp_max);
}

void FWMAP::Term::RemoveUnusedPlanes()
{
	if (fwmap->timestamp_threshold < 0) return;

	int t;
	for (t=0; t<num; t++)
	{
		if (last_accessed[t] < fwmap->timestamp_threshold && num > 1)
		{
			DeletePlane(t --);
		}
	}
}

