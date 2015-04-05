//#include "mex.h"
#include "C:\Program Files\MATLAB\R2008a\extern\include\mex.h"
#include "build_linear_set.h"
#include "build_family_graph.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Check for proper number of arguments. */
	if (nrhs != 2) {
		mexErrMsgTxt("Two inputs required.");
	} else if (nlhs > 2) {
		mexErrMsgTxt("Too many output arguments.");
	}
	
	mexPrintf("Start\n");
	size_t numObjects = mxGetM(prhs[0]);
	size_t dim = mxGetN(prhs[0]);

	double* pos = mxGetPr(prhs[0]);
	vector< vector<double> > sample(numObjects, vector<double> (dim));

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < numObjects; j++) {
			sample[j][i] = *pos;
			//mexPrintf("%d %d %f\n", i, j, sample[j][i]);
			pos++;
		}
	}

	//mexPrintf("-----\n");
	vector<bool> sampleClasses(numObjects, false);
	pos = mxGetPr(prhs[1]);
	for (int i = 0; i < numObjects; i++) {
		if (*(pos + i) > 0.5)
			sampleClasses[i] = true; 
		//mexPrintf("%d %d\n", i, (int)sampleClasses[i]);
	}

	vector<TBitset> algs;

	TLinearSetBuilder builder;
	builder.BuildLinearSet(sample, sampleClasses, algs);
	mexPrintf("Number of algs %d\n", algs.size());

	if (nlhs == 2) {
		plhs[1] = mxCreateLogicalMatrix(algs.size(), numObjects);
		bool *resPtr = mxGetLogicals(plhs[1]);
		for (int i = 0; i < numObjects; i++) {
			for (int j = 0; j < algs.size(); j++) {
				*resPtr = algs[j].Get(i);
				//mexPrintf("%d %d %d\n", i, j, (int)algs[j].Get(i));
				resPtr++;
			}
		}
	}

	mexPrintf("Generating graph\n");
	
	vector< vector<int> > graph;
	BuildFamilyGraph(algs, graph);

	const int t = graph.size();
	plhs[0] = mxCreateCellArray(1, &t);
	for (int i = 0; i < graph.size(); i++) {
		if (graph[i].size() > 0) {
			const int size = graph[i].size();
			mxArray* adjList = mxCreateNumericArray(1, &size, mxINT32_CLASS, mxREAL);
			int* p = reinterpret_cast<int*>(mxGetData(adjList));
			for (int j = 0; j < graph[i].size(); j++)
				*(p + j) = graph[i][j];
		
			mxSetCell(plhs[0], i, mxDuplicateArray(adjList));
		}
	}
		//delete[] listSizes;
}
