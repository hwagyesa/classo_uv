Code for solving the constrained LASSO problem

min\_{U} \| U \|\_1
s.t     \| y\_i -  V' * u\_i \|\_2^2 <= T\_i, i = 1, 2, \dots, p

for some p by n data matrix Y, with rows y\_i, which we
hypothesize factors as Y = U\*V', with U and V having inner
dimension k. 

The problem is solved by solving the KKT conditions, which we
solve using ADMM on the Lagrangian and binary search on the dual
variables (since it suffices to satisfy complementary slackness,
and the problem is convex and assumed strictly feasible). The
Lagrangian minimization problem factors into p separate vector
LASSO problems.
