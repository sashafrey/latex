#pragma once

#include <vector>

using std::vector;
using std::ostream;
using std::exception;

class TBitset {
	static const size_t CellShift = 5;
	static const size_t CellSize = 1 << CellShift;
	static const size_t CellMask = CellSize - 1;
	
	size_t Size;
	vector<size_t> Data;

private:

	void CreateStorage(size_t size) {
		Size = size;
		Data = vector<size_t>(((Size - 1) >> CellShift)  + 1, 0);
	}

public:
	TBitset() {
		Size = 0;
	}

	TBitset(size_t size, bool value = false) {
		CreateStorage(size);

		if (value) {
			for (size_t i = 0; i < Data.size(); i++) {
				Data[i] = 0xFFFFFFFFu;
			}
			for (size_t i = Size; i < Data.size() * CellSize; i++)  {
				Set(i, false);
			}
		}
	}

	TBitset(const vector<bool>& a) {
		CreateStorage(a.size());
		for (size_t i = 0; i < Size; i++) {
			Set(i, a[i]);
		}
	}

	size_t GetSize() const {
		return Size;
	}

	bool Get(size_t pos) const {
		return (Data[pos >> CellShift] & (1 << (pos & CellMask))) != 0; 
	}

	size_t CountOnes() const {
		size_t count = 0;
		for (size_t i = 0; i < Size; i++) {
			if (Get(i)) count++;
		}
		return count;
	}

	void Set(size_t pos, bool val) {
		size_t n = pos >> CellShift;
		size_t r = pos & CellMask;
		Data[n] = (Data[n] ^ (Data[n] & (1 << r))) | (val << r);  
	}

	TBitset Xor(const TBitset& other) {
		if (other.Size != Size)
			throw exception("Bitset sizes doesn't match");
		TBitset res(Size);
		for (size_t i = 0; i < Data.size(); i++)
			res.Data[i] = Data[i] ^ other.Data[i];
		return res;
	}

	TBitset Or(const TBitset& other) {
		if (other.Size != Size)
			throw exception("Bitset sizes doesn't match");
		TBitset res(Size);
		for (size_t i = 0; i < Data.size(); i++)
			res.Data[i] = Data[i] | other.Data[i];
		return res;
	}

	inline friend size_t hash_value(const TBitset& a) {
		return stdext::_Hash_value(a.Data.begin(), a.Data.end());
	}

	friend bool operator <= (const TBitset& a, const TBitset& b) {
		if (a.GetSize() != b.GetSize())
			throw exception("Bitset sizes doesn't match");
		for (size_t i = 0; i < a.Data.size(); i++) {
			if ((a.Data[i] & b.Data[i]) != a.Data[i])
				return false;
		}
		return true;
	}

	friend class CompareByData;

	friend bool operator == (const TBitset& a, const TBitset& b) {
		if (a.GetSize() != b.GetSize())
			return false;
		return a.Data == b.Data;
	}

};

bool CompareByOnesCount(const TBitset& a, const TBitset& b) {
	return a.CountOnes() < b.CountOnes();
}



ostream& operator << (ostream& out, const TBitset& bitset) {
	for (size_t i = 0; i < bitset.GetSize() - 1; i++) {
		out << bitset.Get(i) << " ";
	}
	out << bitset.Get(bitset.GetSize() - 1);
	return out;
}

class CompareByData {
public:
	bool operator () (const TBitset& a, const TBitset& b) const {
		if (a.GetSize() != b.GetSize())
			throw exception("Bitset sizes doesn't match");
		return a.Data < b.Data;
	}
};