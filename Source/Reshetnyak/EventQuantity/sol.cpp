#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>

using namespace std;

const int MaxN = 51;

_int64 C[MaxN][MaxN];
double H[MaxN][MaxN];

int max(int a, int b)
{
	return (a > b) ? a : b;
}

int min(int a, int b)
{
	return (a < b) ? a : b;
}

void CalcC()
{
	for (int i = 0; i < MaxN; i++)
	{
		C[i][0] = 1;
		for (int j= 1; j <= i; j++) C[i][j] = C[i-1][j-1] + C[i-1][j];
	}
}

void CalcH(int l, int L)
{
	int k = L - l;
	for (int m = 0; m <= L; m++)
	{
		int s = max(0, m - k);
		H[m][s] = C[m][s] * C[L - m][l - s];
		for (s = max(0, m - k) + 1; s <= min(l, m); s++)
			H[m][s] = H[m][s - 1] + C[m][s] * C[L - m][l - s];
	}
}

double GetEps(int m, double eta, int l, int L)
{
	int p =  l * (L - l);

	for (int i = 0; i <= l * m; i++)
		if ( H[m][(l * m - i)/L ] / C[L][l] <= eta) return i / (double) p;
	if (m <= L - l) return m / (double) (L - l);
	return (L - m) / (double) l;
}

int main()
{
	

	ifstream inp("input.txt");
	ofstream out("output.txt");

	int l, L, numSteps;
	
	inp >> L >> l >> numSteps;
	int k = L - l;
	
	CalcC();
	CalcH(l, L);
	
	for (int i = 1; i < numSteps; i++)
	{
		
		double eta = (double) i / numSteps;
		out << "eta = " << eta << endl;
		for (int s = 0; s <= l; s++)
		{
			int err = s;
			
			while (l * err < k * (s + l * GetEps(s + err, eta, l, L) ) && err <= k ) err++;
			err--;

			out << "s = " << s << ", err = " << s + err << " ";
		}
		out << endl;
	}
	
	return 0;

}