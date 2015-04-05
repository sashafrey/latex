#pragma once
#include <hash_map>

#include "bitset.h"

//algs отсортирован по числу ошибок
void BuildHasseGraph(const vector<TBitset>& algs, vector< vector<int> >& adjacencyList, int maxEdgeWeight) {
	size_t numAlgs = algs.size();
	vector<size_t> totalError(numAlgs);
	for (int i = 0; i < algs.size(); i++) {
		totalError[i] = algs[i].CountOnes();
	}

	vector<TBitset> connectedVertices(numAlgs, TBitset(numAlgs, false));
	adjacencyList.resize(numAlgs);
	for (int i = numAlgs - 1; i > -1; i--) {
		for (int j = i + 1; j < algs.size() && totalError[j] <= totalError[i] + maxEdgeWeight; j++) {
			if (!connectedVertices[i].Get(j)) {
				if (algs[i] <= algs[j]) {
					adjacencyList[i].push_back(j + 1);
					connectedVertices[i]  = connectedVertices[i].Or(connectedVertices[j]);
					connectedVertices[i].Set(j, true);
				}
			}
		}
	}
}


