
r = 100
c = 1e-6
A =  [-1/(r*c)]
B = [1/(r*c)]
C = [1]
D = [0]
X0 = 0

S = syslin("c",A,B,C)
pas = (5*r*c)

t_final =[]
s_final =[]
for i = 0:10
    t = pas*i:(r*c/2):pas*(i+1)
    if(modulo(i+1,2) == 1)
        u = 1 + 0*t
     else
        u = 0*t 
     end
    s = csim(u,t,S,X0)
    t_final = [t_final t]
    s_final = [s_final s]
    X0 = s(length(s))
end

plot(t_final,s_final)

