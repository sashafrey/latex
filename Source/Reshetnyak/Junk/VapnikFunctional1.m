function VapnikFunctional1
	Lind = [100:2:3000];
	eps = 0.05;
	cnt = 0;
	res = zeros(size(Lind));
	for L = Lind;
		cnt = cnt + 1;
		mvals = [1:L];
		l = L/2;
		svals = l/L * (mvals - (L - l) * eps);
		ind = svals >= 0;
		res(cnt) = (L * (L - 1) + 2) * max( hygecdf(svals(ind), L, mvals(ind), l) );
	end
	figure
	plot(Lind, res, 'r');
end