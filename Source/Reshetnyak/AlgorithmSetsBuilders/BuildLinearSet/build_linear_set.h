#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <hash_set>
#include <math.h>
#include <algorithm>

#include "bitset.h"

using namespace std;

class TLinearSetBuilder {
	stdext::hash_set<TBitset> Cells;
	vector< vector<double> > Sample;
	int Dim;
	int NumPoints;
	int NumAlgs;

	vector<int> CurVertex;

private:

	int GetNumAlgs(int numPoints, int dim) {
		long long res = 0;
		long long cur = 1;

		for (int i = 0; i <= dim; i++) {
			res += cur;
			cur = (cur * (numPoints - i)) / (i + 1);
		}
		return (int) 2 * res;
		cerr << res << "\t" << numPoints << "\t" << dim << endl;
	}

	void Init(const vector< vector<double> >& sample) {
		Sample = sample;
		Dim = sample[0].size();
		NumPoints = sample.size();
		NumAlgs = GetNumAlgs(NumPoints, Dim);
	}

	void SolveLinearSystem(vector< vector<double> > a, vector<double> b, vector<double>& res) {
		int n = a.size();
		for (int i = 0; i < n; i++) {
			int bestRow  = i;
			for (int j = i + 1; j < n; j++) {
				if (fabs(a[j][i]) > fabs(a[bestRow][i])) bestRow = j;
			}
			if (a[bestRow][i] == 0) continue;
			
			if (i != bestRow) {
				a[i].swap(a[bestRow]);
				swap(b[i], b[bestRow]);
			}

			for (int j = i + 1; j < n; j++) {
				double p = a[j][i] / a[i][i];
				b[j] -= b[i] * p;
				for (int h = i + 1; h < n; h++)
					a[j][h] -= p * a[i][h];
			}
		}

		//res = vector<double> (n, 0);
		for (int i = b.size() - 1; i > -1; i--) {
			res[i] = b[i] / a[i][i];
			for (int j = i + 1; j < a.size(); j++)
				res[i] -= res[j] * a[i][j] / a[i][i];
		}
	}

	void GetHyperplane(const vector<int>& vertex, vector<double>& hyperplane) {
		vector< vector<double> > points(Dim, vector<double> (Dim, 0) ); 
		for (int i = 0; i < Dim; i++) {
			for (int j = 0; j < Dim; j++)
				points[i][j] = Sample[vertex[i]][j];
		}

		vector<double> shifts(Dim, 1);
		hyperplane = vector<double> (Dim + 1);
		SolveLinearSystem(points, shifts, hyperplane);
		hyperplane[Dim] = -1;
	}

	bool ClassifyPoint(const vector<double>& hyperplane, const vector<double>& point) {
		double score = hyperplane[Dim];
		for (int i = 0; i < Dim; i++)
			score += hyperplane[i] * point[i];
		return score > 0;
	}

	void ClassifyPoints(const vector<double>& hyperplane, TBitset& res) {
		for (int i = 0; i < NumPoints; i++) {
			res.Set(i, ClassifyPoint(hyperplane, Sample[i]));
		}
	}

	void GenerateAllVariants(const vector<int>& vertex, TBitset cell) {
		TBitset ones(NumPoints, true);
		for (int i = 0; i < (1 << Dim); i++) {
			for (int j = 0; j < Dim; j++) {
				if (i & (1 << j)) {
					cell.Set(vertex[j], true);
				} else {
					cell.Set(vertex[j], false);
				}
			}
			Cells.insert(cell);
			Cells.insert(cell.Xor(ones));

		}
	}

	void GenerateCells(const vector<int>& vertex) {
		vector<double> hyperplane;
		TBitset cell(NumPoints);
		GetHyperplane(vertex, hyperplane);
		ClassifyPoints(hyperplane, cell);
		GenerateAllVariants(vertex, cell);
	}

	void RecursivelyEnumerateVertices(int used) {
		if (used == Dim) {
			GenerateCells(CurVertex);
			return ;
		}
		int startPos = 0;
		if (used > 0)
			startPos = CurVertex[used - 1] + 1; 
		for (int i = startPos; i < NumPoints; i++) {
			CurVertex[used] = i;
			RecursivelyEnumerateVertices(used + 1);	
		}
	}
		
	void EnumerateAllVertices() {
		CurVertex = vector<int>(Dim, 0);
		RecursivelyEnumerateVertices(0);
	}

	void GenerateAlgorithms(const vector<bool>& sampleClasses, vector<TBitset>& algs) {
		TBitset ones(NumPoints, true);
		//if (Cells.size() != NumAlgs)
		//	throw exception("Incorrect number of algorithms");
		algs = vector<TBitset>(Cells.size(), TBitset(sampleClasses));
		int pos = 0;
		for (stdext::hash_set<TBitset>::const_iterator hi = Cells.begin(); hi != Cells.end(); ++hi) {
			algs[pos] = algs[pos].Xor(*hi);
			pos++;
		}
		sort(algs.begin(), algs.end(), CompareByOnesCount);
	}

public:
	void BuildLinearSet(const vector< vector<double> >& sample, const vector<bool>& sampleClasses, vector<TBitset>& algs) {
		Init(sample);
		EnumerateAllVertices();
		GenerateAlgorithms(sampleClasses, algs);
	}	
};
