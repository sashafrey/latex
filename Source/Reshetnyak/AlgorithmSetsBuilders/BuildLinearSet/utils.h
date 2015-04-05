#include "bitset.h"

void GetUniqueAlgorithms(vector<TBitset>& algs) {
	sort(algs.begin(), algs.end(), CompareByData());
	vector<TBitset>::iterator newEnd = unique(algs.begin(), algs.end());
	algs.resize(newEnd - algs.begin());
}

void GetFirstLevels(vector<TBitset>& algs, int numLevels) {
	sort(algs.begin(), algs.end(), CompareByOnesCount);
	for (int lastAlg = 0; lastAlg < algs.size(); lastAlg++) {
		if (algs[0].CountOnes() + numLevels <= algs[lastAlg].CountOnes()) {
			algs.resize(lastAlg);
			break;
		}
	}
}

void AddPseudoAlgorithms(vector<TBitset>& algs, int numLevels, int maxEdgeWeight) {
	//Массив algs - отсортирован по числу ошибок!!!
	vector<size_t> totalError(algs.size());
	for (int i = 0; i < algs.size(); i++) {
		totalError[i] = algs[i].CountOnes();
	}
    numLevels += totalError[0];
	size_t numAlgs = algs.size();
	for (size_t i = 0; i < numAlgs && totalError[i] < numLevels; i++) {
		for (size_t j = i + 1; j < numAlgs && totalError[j] < numLevels && totalError[j] <= totalError[i] + maxEdgeWeight; j++) {
			if (algs[i].Xor(algs[j]).CountOnes() <= maxEdgeWeight && !(algs[i] <= algs[j])) {
				if (algs.capacity() == algs.size()) {
					algs.reserve(2 * algs.capacity());
				}
				algs.push_back(algs[i].Or(algs[j]));
			}
		}
	}

   GetUniqueAlgorithms(algs);
}