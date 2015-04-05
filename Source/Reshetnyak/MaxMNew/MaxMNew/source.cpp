#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>

using namespace std;

const int MAXN = 51;

long long C[MAXN][MAXN];
long long H[MAXN][MAXN];
//ong long count;
int l, L, numSteps;

ifstream inp("input.txt");
ofstream out("output.txt");

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
	for (int i = 0; i < MAXN; i++)
		for (int j = 0; j < MAXN; j++) C[i][j] = 0;

	for (int i = 0; i < MAXN; i++)
	{
		C[i][0] = 1;
		for (int j = 1; j <= i; j++) C[i][j] = C[i-1][j-1] + C[i-1][j];
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
		{
			H[m][s] = H[m][s - 1] + C[m][s] * C[L - m][l - s];
		}
	}

	/*
	for (int m = 0; m <= L; m++)
	{
		out << m << ") " ;
		for (int s = max(0, m - k); s <= min(l, m); s++)
		{
			out << H[m][s] << " "; 
		}
		out << endl;
	}
*/
}



int InverseH(int s, double delta)
{
	int m;
	for (m = s; m <= L; m++)
	{
		if (H[m][s] <= delta * C[L][l]) return m;
	}

	return m;
}

int main()
{
	
	inp >> L >> l >> numSteps;
	int k = L - l;
	
	CalcC();
	CalcH(l, L);
	//count = C[L][l];
	
	// Check my proof for some delta and m
	/*
	for (int m = 0; m <= L; m++)
	{
		for (double delta = 0.01; delta < 0.205; delta += 0.01)
		{
			for (int s = 0; s <= min(m, l); s++)
			{
				int mEstimate = InverseH(s, delta);
				out << "m = " << m << " delta =  " << delta << " s = " << s << " mEstimate = " << mEstimate << endl;
			}
		}
	}
	*/

	for (double delta = 0.005; delta < 0.205; delta += 0.005)
	{
		for (int s = 0; s <= l; s++)
		{
			int mEstimate = InverseH(s, delta);
			out <<  " delta =  " << delta << " s = " << s << " mEstimate = " << mEstimate << endl;
		}
	}
	return 0;

}