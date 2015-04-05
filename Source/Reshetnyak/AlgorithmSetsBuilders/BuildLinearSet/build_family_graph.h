#pragma once
#include <xhash>
#include <hash_map>


#include "bitset.h"

void BuildFamilyGraph(vector<TBitset>& algs, vector< vector<int> >& adjacencyList) {
	stdext::hash_map<TBitset, int, stdext::hash_compare<TBitset, CompareByData > > algNumbers;
	for (int i = 0; i < algs.size(); i++) {
		algNumbers[algs[i]] = i + 1;
	}

	adjacencyList.resize(algs.size());
	for (int i = 0; i < algs.size(); i++) {
		for (size_t pos = 0; pos < algs[i].GetSize(); pos++) {
			bool val = algs[i].Get(pos);
			algs[i].Set(pos, !val);
			stdext::hash_map<TBitset, int>::const_iterator hi = algNumbers.find(algs[i]);
			if (hi != algNumbers.end()) {
				adjacencyList[i].push_back(hi->second);
			}
			algs[i].Set(pos, val);
		}
	}
}


