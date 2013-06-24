import numpy as np

def hdot(A,B):
    (a11,a12,a13,a21,a22,a23) = A
    (b11,b12,b13,b21,b22,b23) = B

    AB2 = a21*b21 + a22*b22 + a23*b23

    return [\
    a11*b11 - a21*b21 + AB2,
    a12*b12 - a22*b22 + AB2,
    a13*b13 - a23*b23 + AB2,
    a23*b22 + a12*b21 + b13*a21,
    a21*b23 + a13*b22 + b11*a22,
    a22*b21 + a11*b23 + b12*a23\
    ]


def hddot(A,B):
    (a11,a12,a13,a21,a22,a23) = A
    (b11,b12,b13,b21,b22,b23) = B

    AB1 = a11*b11 + a12*b12 + a13*b13
    AB2 = a21*b21 + a22*b22 + a23*b23

    return AB1 + 2*AB2


