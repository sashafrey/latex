#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <hash_set>
#include <math.h>
#include <algorithm>

#include "bitset.h"
#include "utils.h"

using namespace std;


class TConjunctionSetSimpleBuilder {
	vector< vector<double> > Sample;
	vector< vector< pair<double, int> > > SortedSample;

	int Dim;
	int SampleSize;
	
	TBitset SampleClasses;
	TBitset Ones;

private:

	
	void Init(const vector< vector<double> >& sample, const vector<bool>& sampleClasses) {
		Sample = sample;
		SampleClasses = TBitset(sampleClasses);
		Dim = sample[0].size();
		SampleSize = sample.size();
		Ones = TBitset(SampleSize, true);
	}

	void ConstructSortedSample() {
		SortedSample = vector< vector< pair<double, int> > >(Dim, vector< pair<double, int> >(SampleSize));
		for (size_t i = 0; i < Dim; i++) {
			for (size_t j = 0; j < SampleSize; j++) {
				SortedSample[i][j].first = Sample[j][i];
				SortedSample[i][j].second = j;
			}
			sort(SortedSample[i].begin(), SortedSample[i].end());
		}
	}

	template <class TIterator>
	void MoveThreshold(TBitset curAlg, int curTerm, vector<TBitset>& algs, TIterator begin, TIterator end) {
		int prevChanges = 0;
		double prevVal = begin->first;
		for (TIterator it = begin; it != end; ++it) {
			if (it->first != prevVal) {
				prevVal = it->first;
				if (prevChanges > 0) {
					RecursivelyGenerateAlgorithms(curAlg, curTerm + 1, algs);
				}
				prevChanges = 0;
			}
			if (curAlg.Get(it->second)) {
				prevChanges++;
				curAlg.Set(it->second, 0);
			}
		}
	}

	void RecursivelyGenerateAlgorithms(const TBitset& curAlg, int curTerm, vector<TBitset>& algs) {
		if (curTerm == Dim) {
			/*if (algs.capacity() < algs.size() + 1 + InvertAlgorithms) {
				algs.reserve(2 * algs.capacity() + 1);
			}*/
			algs.push_back(SampleClasses.Xor(curAlg));
			return ;
		}
		RecursivelyGenerateAlgorithms(curAlg, curTerm + 1, algs);
		MoveThreshold(curAlg, curTerm, algs, SortedSample[curTerm].begin(), SortedSample[curTerm].end());
		MoveThreshold(curAlg, curTerm, algs, SortedSample[curTerm].rbegin(), SortedSample[curTerm].rend());
	
	}

	size_t EstimateNumberOfAlgs() {
		size_t res = 1;
				for (size_t i = 0; i < Dim; i++) {
			size_t thresholds = 1;
			for (size_t j = 1; j < SortedSample[i].size(); j++) {
				if (SortedSample[i][j].first > SortedSample[i][j - 1].first) 
					thresholds++;
			}
			res *= 2 * thresholds;
		}
		res = min(res, 1u<<22);
		return res;
	}



public:
	void BuildConjunctionSet(const vector< vector<double> >& sample, const vector<bool>& sampleClasses, vector<TBitset>& algs, int numLevels) {
		Init(sample, sampleClasses);
		ConstructSortedSample();
		algs.reserve(EstimateNumberOfAlgs());
		algs.push_back(SampleClasses);
		RecursivelyGenerateAlgorithms(Ones, 0, algs);
		cerr << "Number of generated algs=" << algs.size() << endl;
		GetUniqueAlgorithms(algs);
		GetFirstLevels(algs, numLevels);
	}	
};
