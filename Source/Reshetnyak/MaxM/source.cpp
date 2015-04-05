#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>

using namespace std;


const int MAXN = 50;
const int MAXL = 50;

int main()
{
	
	
	__int64 C[MAXN + 1][MAXL + 1];
	
	
	fill( &C[0][0], &C[MAXN][MAXL] + 1, 0);
	C[0][0] = 1;
	for (int i = 1; i <= MAXN; i++)
	{
		C[i][0] = 1;
		for (int j = 1; j <= i; j++)
			C[i][j] = C[i-1][j-1] + C[i-1][j];
    }

	int N, maxl, eps; 
	cin >> N >> maxl >> eps;
	
	ofstream out("data.txt");
	for (int l = 1; l <= maxl; l++)
	{
		__int64 max_val = 0; 
		__int64 cur_f;
		__int64 d_val = 0;
		int max_m1 = 0;
		int max_m2 = 0;
		for (int m1 = 0; m1 <= N; m1++)
		{
			for (int m2 = 0; m2 <= N - m1; m2++)
			{
				cur_f = 0;
				int s = max(1, eps + m2 - m1);
				for (int i = s; i <= l; i++)
					for (int j = 0; 2 * j + s <= i; j++)
						cur_f += C[N - m1 - m2][l - i] * C[m1][j] * C[m2][i - j];
			   //	if (maxl == l and eps + m2- m1 == 1) cout << m1 << " " << m2 << " " << cur_f/(double) C[N][l] << endl;
				if (l == maxl && m2 == m1 - eps + 1) out << m1 << " "<< (double)cur_f / C[N][l] << endl; 
 				if (cur_f > max_val)
				{	
					max_val = cur_f;
					max_m1 = m1;
					max_m2 = m2;
				}
				if ( (m1 + m2 == N) or (m1 + m2 == N - 1) )
					if (cur_f > d_val) d_val = cur_f; 
			}
		}
		//cout << l << ": " << cur_f << " " << ( 0.5 * cur_f ) / C[N][l]  << endl;
		cout << "l: " << l << " m1: " << max_m1 << " m2: " << max_m2 << " val: " << max_val;
		cout << " p: " << max_val/(double)C[N][l] << " dp:" << d_val /(double)C[N][l] << endl;
		//cout << "Minimum at " << min_num << " and equals " << min_val << endl;
	}
	
	
	/*
	//Check another Hypotesis
	int maxM, eps;
	cin >> maxM >> eps;
	for (int M = 1; M <= maxM; M++)
	{
		for (int k = 1; k <= M - 1; k++)
		{
			int max_num = 0;
			__int64 max_val = 0;
			for (int m1 = 0; m1 < M; m1++)
			{
				int s = max(1, M - 2* m1 + eps);
				__int64 cur_f = 0;
				for (int i = 0; 2 * i + s <= k; i++)
					cur_f += C[m1][i]*C[M-m1][k-i];
				//cout << m1 << " " << cur_f << endl;
				if (cur_f > max_val)
				{
					max_val = cur_f;
					max_num = m1;
				}
			}
			if ( (2 * max_num != (M + eps - 1) ) and  (2 * max_num != (M + eps) )) cout << "Fail! M:"  << M << " k:" << k << " m1:" <<max_num << " v:" << max_val<< endl;
		}
	}
	*/
/*	//With eps
	
	int eps;
	//cin >> N >> maxl >> eps;
	cin >> eps;
	for (int N = 2; N <= 50; N++)
	{	
		for (int l = 1; l <= N; l++)
		{
			__int64 max_val = 0; 
			__int64 cur_f;
			int max_num = 0;
			for (int m = 0; m <= N/2; m++)
			{
				int b = min(l, 2 * m - 1);
				cur_f = 0;
				for (int i = eps; i <= b; i++)
					for (int j = 0; 2 * j + eps <= i; j++)
						cur_f += C[N - 2*m][l - i] * C[m][j] * C[m][i - j];
		
			   // cout << m << ": " << cur_f << endl;
				if (cur_f > max_val)
				{
					max_val = cur_f;
					max_num = m;
				}
			}
			double p = (double)max_val / C[N][l];
			if ( (l - eps) % 2 )
				if ( (p > 0.5 - (0.5 * eps) /(l + 1) ) or (p < 0.5 - (0.5 * (eps + 1) * C[N/2][l/2] * C[N/2][l/2] / (double)C[N][l]) ) ) 
			cout << "Fail " << N << " " << l << ": " << max_num << " "  << max_val << " " << p << endl;
			//cout << l << ": " << cur_f << " " << ( 0.5 * cur_f ) / C[N][l]  << endl;
		}
	}
	*/
	//*/
	/*CheckHypotesis
	for (int n = 2; n <= 50; n++)
	{
		for (int l = 1; l <= n; l++)
		{
			__int64 min_val = (__int64) 1 << 62; 
			__int64 cur_f;
			int min_num = 0;
			for (int m = 0; m <= n/2; m++)
			{
				int b = min(l/2, m);
				cur_f = 0;
				for (int i = 0; i <= b; i++)
					cur_f += C[n - 2*m][l - 2*i] * C[m][i] * C[m][i];
		
			//cout << m << ": " << cur_f << endl;
				if (cur_f < min_val && cur_f != 0)
				{
					min_val = cur_f;
					min_num = m;
				}
			}
			if (l%2 == 0 and (l + 1) * min_val < C[n][l]) 
			//if ( (l%2 == 0) and (min_val > C[n/2][l/2] * C[n/2][l/2] ) )
			cout << "Fail " << n << " " << l << " " << min_num <<" " << min_val << " " << C[n][l] << endl;
		}
		
	}
    */
	cout << "!!!" << endl;
	return 0;
}

