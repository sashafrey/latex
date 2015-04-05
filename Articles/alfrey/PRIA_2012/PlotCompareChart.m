%=====================================================
%begin{Liver disorders, RSA}
a = [0,-0.0006,0.0041,0.0078,0.0078,0.0078,0.0017,0.0567;
1,0.0104,0.0273,0.0391,0.0391,0.0391,0.0104,0.0567;
2,0.0232,0.062,0.0703,0.0703,0.0703,0.0266,0.0567;
3,0.0272,0.0736,0.0859,0.0859,0.0859,0.0272,0.0567;
4,0.0319,0.0968,0.1016,0.1016,0.1016,0.0319,0.0567;
5,0.0382,0.1026,0.1016,0.1016,0.1016,0.0382,0.0567;
6,0.0411,0.1083,0.1172,0.1016,0.1016,0.0428,0.0567;
7,0.044,0.113,0.1172,0.1172,0.1172,0.0448,0.0567;
8,0.048,0.1141,0.1172,0.1172,0.1172,0.0472,0.0567;
9,0.0469,0.1193,0.1328,0.1172,0.1172,0.0486,0.0567;
10,0.0498,0.1199,0.1328,0.1172,0.1328,0.0498,0.0567;
11,0.0515,0.12,0.1328,0.1328,0.1328,0.0504,0.0567;
12,0.051,0.1211,0.1328,0.1328,0.1328,0.051,0.0567;
13,0.0538,0.1222,0.1328,0.1328,0.1328,0.0527,0.0567;
14,0.0541,0.124,0.1328,0.1328,0.1328,0.0527,0.0567;
15,0.0527,0.1257,0.1328,0.1328,0.1328,0.0538,0.0567;
16,0.0527,0.1251,0.1328,0.1328,0.1328,0.0527,0.0567;
17,0.0553,0.1257,0.1484,0.1328,0.1328,0.0527,0.0567;
18,0.0544,0.1257,0.1484,0.1328,0.1328,0.0544,0.0567;
19,0.0538,0.1257,0.1484,0.1328,0.1484,0.0538,0.0567;
20,0.0544,0.1257,0.1484,0.1484,0.1484,0.0527,0.0567;
21,0.0541,0.1257,0.1484,0.1484,0.1484,0.0538,0.0567;
22,0.055,0.1257,0.1484,0.1484,0.1484,0.0532,0.0567;
23,0.0533,0.1257,0.1484,0.1484,0.1484,0.0538,0.0567;
24,0.0527,0.1257,0.1484,0.1484,0.1484,0.0538,0.0567;
25,0.0539,0.1257,0.1484,0.1484,0.1484,0.0533,0.0567;
26,0.0533,0.1257,0.1484,0.1484,0.1484,0.0538,0.0567;
27,0.0544,0.1257,0.1641,0.1484,0.1484,0.0561,0.0567;
28,0.0567,0.1263,0.1641,0.1484,0.1484,0.0547,0.0567;
29,0.0547,0.1257,0.1641,0.1641,0.1641,0.0567,0.0567;
30,0.0555,0.1257,0.1641,0.1641,0.1641,0.0555,0.0567;
31,0.0561,0.1257,0.1641,0.1641,0.1641,0.055,0.0567;
32,0.0567,0.1257,0.1641,0.1641,0.1641,0.0553,0.0567;
33,0.0556,0.1257,0.1641,0.1641,0.1641,0.0578,0.0567;
34,0.0555,0.1251,0.1641,0.1641,0.1641,0.0561,0.0567;
35,0.0564,0.1257,0.1641,0.1641,0.1641,0.057,0.0567;
36,0.0567,0.1251,0.1797,0.1641,0.1641,0.0567,0.0567;
37,0.0573,0.1257,0.1797,0.1641,0.1641,0.0567,0.0567];

x = a(:, 1);
btmIC = a(:, 2);
rnd = a(:, 3);
bool = a(:, 4);
hasse = a(:, 5);
hasseSimple = a(:, 6);
btm = a(:, 7);
all = a(:, 8);

plot(x, all, 'k', ...
     x, btm, 'o-k', ...
     x, btmIC, '*-k', ...
     x, hasse, 'd-k', ...
     x, hasseSimple, 'x-k', ...
     x, bool, '--k', ...
     x, rnd, ':k' ...
     );
%legend('Monte-Karlo, A', 'Monte-Karlo, A_r', 'Monte-Karlom A^*_r', 'Hasse bound', 'No upper con. boud', 'Bool bound');
xlabel('r');
ylabel('Eps at 50% quantile');

return;

%end{Liver disorders, RSA}
%=====================================================
%begin{Echocardiogramm, RSA}

a=[0,0.027,0.027,0.0234,0.0234,0.0234,0.027,0.0622;
1,0.0541,0.0541,0.0547,0.0547,0.0547,0.0541,0.0622;
2,0.0541,0.0541,0.0547,0.0547,0.0547,0.0541,0.0622;
3,0.0541,0.0784,0.0859,0.0859,0.0859,0.0541,0.0622;
4,0.0568,0.0811,0.0859,0.0859,0.0859,0.0541,0.0622;
5,0.0649,0.0811,0.1016,0.0859,0.1016,0.0554,0.0622;
6,0.0622,0.0811,0.1016,0.0859,0.1016,0.0622,0.0622;
7,0.0811,0.0811,0.1328,0.1016,0.1016,0.0622,0.0622;
8,0.0811,0.0811,0.1328,0.1016,0.1016,0.0568,0.0622;
9,0.0811,0.0811,0.1641,0.1016,0.1016,0.0581,0.0622;
10,0.0811,0.0811,0.1641,0.1016,0.1016,0.0568,0.0622;
11,0.0811,0.0811,0.1641,0.1016,0.1016,0.073,0.0622;
12,0.0811,0.0946,0.1953,0.1016,0.1328,0.0703,0.0622;
13,0.0811,0.1,0.1953,0.1016,0.1328,0.0608,0.0622;
14,0.0811,0.0838,0.1953,0.1016,0.1328,0.0649,0.0622;
15,0.0811,0.1081,0.2109,0.1328,0.1641,0.0622,0.0622;
16,0.0811,0.1081,0.2422,0.1328,0.1641,0.0608,0.0622;
17,0.1014,0.1297,0.2422,0.1641,0.1641,0.0541,0.0622;
18,0.1027,0.1351,0.2422,0.1641,0.1953,0.0568,0.0622;
19,0.1081,0.1568,0.2734,0.1953,0.2109,0.0554,0.0622];

x = a(:, 1);
btmIC = a(:, 2);
rnd = a(:, 3);
bool = a(:, 4);
hasse = a(:, 5);
hasseSimple = a(:, 6);
btm = a(:, 7);
all = a(:, 8);

plot(x, all, 'k', ...
     x, btm, 'o-k', ...
     x, btmIC, '*-k', ...
     x, hasse, 'd-k', ...
     x, hasseSimple, 'x-k', ...
     x, bool, '--k' ...
     );
legend('Monte-Karlo, A', 'Monte-Karlo, A_r', 'Monte-Karlom A^*_r', 'Hasse bound', 'No upper con. boud', 'Bool bound');
xlabel('r');
ylabel('Eps at 50% quantile');

%end{Echocardiogramm, RSA}
%=====================================================
