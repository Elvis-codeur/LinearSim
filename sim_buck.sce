
l = 1e-2
r = 1
c = 1e-2

A =[[0, -1/l];
    [1/c, -1/(r*c)]
    ]
B =[[1/l, 0];[0, 0]]

Vs = [12, 0]
C = [1, 1]
X0 = [0, 0]

S = syslin("c",A,B,C)
t = 0:1e-3:1

u = [12 + 0*t;0*t]

s = csim(u,t,S)
plot(t,s)
