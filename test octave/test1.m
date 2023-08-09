pkg load signal
Num = [ 890, 4600]
Den = [1, 53]

H = tf(Num,Den)
bode(H)
