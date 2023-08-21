clear all
close

Num = poly([6 5 1],"s","coeff")
w0 = 0.1
z = 0.7
Den = poly([6 11 6 1],"s","coeff")
H = syslin("c",Num/Den)

//bode(H)

t=0:0.1:10;
u = sin(10*t);
gs1=csim("step",t,H);
plot2d(t,gs1)

disp(Num)
disp(Den)

disp(tf2ss(H))

