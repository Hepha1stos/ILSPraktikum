# Musterloesung zu Versuch 2, Aufgabe 1
import numpy as np 
import itertools

def get_phi_polyD1(m):   # return list of 1D polynomial basis functions phi_j(x) of degree m
    phi = [lambda x,n=n: x[0]**n for n in range(m+1)]        # generate list of basis functions phi_n(x):=x^n for n=0,1,...,m
    return lambda x: np.array([phi_j(x) for phi_j in phi])   # return function generating feature vector phi(x) for input x 

def get_phi_polyD2(m):   # return list of 2D polynomial basis functions phi_j(x0,x1) of degree m
    phi = []
    for n in range(m+1):            # loop over total degree n=0,1,2,...,m
        for n0 in range(n+1):       # loop over all n0 (and n1:=n-n0) with n0+n1=n
            n1=n-n0
            phi+=[lambda x,n0=n0,n1=n1: x[0]**n0 * x[1]**n1]    # add basis function x0^n0 x1^n1
    return lambda x: np.array([phi_j(x) for phi_j in phi])   # return function generating feature vector phi(x) for input x 

def get_phi_polyD3(m):   # return list of 3D polynomial basis functions phi_j(x0,x1,x2) of degree m
    phi = []
    for n in range(m+1):            # loop over total degree n=0,1,2,...,m
        for n0 in range(n+1):       
            for n1 in range(n - n0 + 1):
                n2 = n - n0 - n1
                phi += [lambda x, n0=n0, n1=n1, n2=n2: x[0]**n0 * x[1]**n1 * x[2]**n2]  # x0^n0 * x1^n1 * x2^n2
    return lambda x: np.array([phi_j(x) for phi_j in phi])   # return function generating feature vector phi(x) 

def get_phi_poly(d,m):   # return list of all polynomial basis functions for input dimensionality d and polynomial degree m
    phi = []  # init list of basis functions
    list_ni = [ni for ni in itertools.product(range(m+1), repeat=d) if sum(ni) <= m]  # alle Tupel (n1,...,nd) mit sum <= m
    list_ni_sum = np.array([np.sum(ni) for ni in list_ni], 'int')
    idx_list = np.array(range(len(list_ni_sum)))

    for n in range(m+1):
        idx_list_n = idx_list[list_ni_sum == n]
        for i in idx_list_n:
            ni = list_ni[i]
            phi += [lambda x, ni=ni: np.prod([x[j]**ni[j] for j in range(d)])]  # Basisfunktion x1^n1 * x2^n2 * ... * xd^nd
    return lambda x: np.array([phi_j(x) for phi_j in phi])   # return function generating feature vector phi(x)


def evaluate_linear_model(W, phi, x):  # return linear model function y=w.T*phi(x) or y=W*phi(x)
    phi_x = phi(x)
    y = W @ phi_x  # Matrix-Vektor-Produkt
    return y


# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    print("\n*** (i) Test implementations for specific input dimensionality ***")
    phi_poly1D = get_phi_polyD1(5)
    phi_poly2D = get_phi_polyD2(4)
    phi_poly3D = get_phi_polyD3(3)
    print("phi_poly1D([2])=",phi_poly1D([2]))
    print("phi_poly2D([1,2])=",phi_poly2D([1,2]))
    print("phi_poly3D([1,2,3])=",phi_poly3D([1,2,3]))

    print("\n*** (ii) Test general implementation (should yield same results as before in (i)!) ***")
    phi_poly1D_ = get_phi_poly(1,5)
    phi_poly2D_ = get_phi_poly(2,4)
    phi_poly3D_ = get_phi_poly(3,3)
    phi_poly4D_ = get_phi_poly(4,3)
    print("phi_poly1D_([2])=",phi_poly1D_([2]))
    print("phi_poly2D_([1,2])=",phi_poly2D_([1,2]))
    print("phi_poly3D_([1,2,3])=",phi_poly3D_([1,2,3]))
    print("phi_poly4D_([1,2,3,4])=",phi_poly4D_([1,2,3,4]))
    print("len(phi_poly4D_)=",len(phi_poly4D_([1,2,3,4])))
    
    print("\n*** (iii) Test linear model ***")
    arr_1_35 = np.array(range(35),'float')+1.0
    w = np.divide(arr_1_35+1.0,arr_1_35)           # define weight vector w for 1D linear model
    W = np.array([w,np.flipud(w)])                 # define weight matrix W for 2D linear model
    x=[1,2,3,4]              # input vector
    print("y=w.T*phi(x)=",evaluate_linear_model(w,phi_poly4D_,x))
    print("y=W*phi(x)=",evaluate_linear_model(W,phi_poly4D_,x))



    p1 = get_phi_poly(1,4)
    p2 = get_phi_poly(2,3)
    p3 = get_phi_poly(3,2)
    p4 = get_phi_poly(4,3)
    print("p1: ", p1([2]))
    print("p2: ", p2([1,2]))
    print("p3: ", p3([1,2,3]))
    print("p4: ", p4([1,2,3,4]))