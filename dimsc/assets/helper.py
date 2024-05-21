import numpy as np
np.set_printoptions(suppress=True)

###
### Proprietes du pli
###

# AS4D/9310
E1 = 133.86e3
E2 = E3 = 7.706e3
G12 = G13 = 4.306e3
G23 = 2.76e3
v12 = v13 = 0.301
v23 = 0.396

# matrice de souplesse
S = np.array([[1.0/E1, -v12/E1, -v13/E1   , 0     , 0     , 0]       ,
               [-v12/E1, 1.0/E2 , -v23/E2, 0      , 0      , 0]       ,
               [-v13/E1, -v23/E2, 1.0/E3 , 0      , 0      , 0]       ,
               [0      , 0      , 0      , 1.0/G23, 0      , 0]       ,
               [0      , 0      , 0      , 0      , 1.0/G13, 0]       ,
               [0      , 0      , 0      , 0      , 0      , 1.0/G12]]
             )

# matrice de rigidites reduites
Q = np.linalg.inv( np.array([[S[0,0], S[0,1], 0],
                             [S[1,0], S[1,1], 0], 
                             [     0,      0, S[5,5]]]))

Q1 = np.linalg.inv( np.array( [[S[3,3], 0], 
                              [0,      S[4,4]]]))

###
### Definition de l'empilement
###

# [0/90/45/-45]s
angles = [0, np.pi/2.0, np.pi/4.0, -np.pi/4.0, -np.pi/4.0, np.pi/4, np.pi/2.0, 0] 
thicknesses = [1, 1, 1, 1, 1, 1, 1, 1]

###
### Calcul des matrices ABD
###

h = np.sum(thicknesses)
zkm = -0.5 * h

A = np.zeros_like(Q)
B = np.zeros_like(Q)
D = np.zeros_like(Q)
H = np.zeros_like(Q1)

for k in range(len(angles)):
    theta = angles[k]
    c = np.cos(theta)
    s = np.sin(theta)
    
    M = np.array([[c*c, s*s, -2*c*s],
                  [s*s, c*c,  2*c*s],
                  [c*s, -c*s, c*c-s*s]])
    Qp = np.dot(M, np.dot(Q, M.transpose()))
    
    M1 = np.array([[c, s],
                   [-s, c]])
    Q1p = np.dot(M1, np.dot(Q1, M1.transpose()))

    zk = zkm + thicknesses[k]
    A += (zk - zkm) * Qp
    B += 0.5 * (zk**2 - zkm**2) * Qp
    D += 0.3333333333 * (zk**3 - zkm**3) * Qp
    H += 0.8333333333 * (zk - zkm) * Q1p
    zkm += thicknesses[k]

print("A: ")
print(A)

print("B: ")
print(B)

print("D: ")
print(D)

print("H:")
print(H)

###
### Calcul du materiau ortho. equivalent
###

# en membrane
Exx = (A[0,0]*A[1,1] - A[0,1]**2)/(h*A[1,1])
Eyy = (A[0,0]*A[1,1] - A[0,1]**2)/(h*A[0,0])
Gxy = A[2,2]/h
v = A[0,1]/A[1,1]
print("Materiau ortho. membrane")
print('Ex: ', Exx, 'Ey: ', Eyy, 'Gxy: ', Gxy, 'nu: ', v)

# en flexion
Exx = 12.0*(D[0,0]*D[1,1] - D[0,1]**2)/(h**3*D[1,1])
Eyy = 12.0*(D[0,0]*D[1,1] - D[0,1]**2)/(h**3*D[0,0])
Gxy = 12.0*D[2,2]/h**3
v = D[0,1]/D[1,1]
print("Materiau ortho. flexion")
print('Ex: ', Exx, 'Ey: ', Eyy, 'Gxy: ', Gxy, 'nu: ', v)

