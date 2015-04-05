L = 400;
l = 200;
m = 40;
delta = 0.1;
sMax = 10;
s0 = 20;
S = repmat([0:l], l + 1, 1);
H = hygecdf( S, L, S' , l);
H([s0+1: l + 1], sMax).^2 
plot( nchoosek(L, l) * H([s0+1: l + 1], sMax).^2 );