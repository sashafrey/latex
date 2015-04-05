function StudyHypergeometricCdfFunction(L, l, eps)
	
	if nargin < 1
		L = 1000
	end
	if nargin < 2
		l = floor(L / 2);
	end
	if nargin < 3
		eps = 0.1;
	end
	
	svals = floor( l/L * ([0:L] - eps * (L - l) ) );
	
	hygecdf(svals, L, [0:L], l) 
	plot( [0:L], hygecdf(svals, L, [0:L], l) );
		
end