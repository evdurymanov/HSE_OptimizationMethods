import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
rec = 1000000000
sys.setrecursionlimit(rec)



def func_1(x_1, x_2, n):
    return 100 * ((x_1 - x_2) ** 2) + 5 * 5

def func_1_dot(x_1, x_2, n):
    pass

def func_2(x_1, x_2):
    return 100 * (x_2 - x_1 ** 2) ** 2 + 5 * (1 - x_1) ** 2

def func_2_grad(x_1, x_2):
    return (10 * (40 * x_1 ** 3 - 40 * x_1 * x_2 + x_1 - 1), 200 * (x_2 - x_1 ** 2))



def golden(f, b, a, c1, c2, c3, c4, epsilon = 10 ** (-12)):
    phi = (1 + (5 ** 1 / 2)) / 2
    c, d = b - (b - a) / phi, a + (b - a) / phi
    while abs(b - a) > epsilon:
        if f(c, c1, c2, c3, c4) < f(d, c1, c2, c3, c4):
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi

    return (a + b) / 2





def grad_1(f, fgrad, x_1, x_2, x_s, y_s, case):

    start = f(x_1, x_2)

    fdot1, fdot2 = fgrad(x_1, x_2)
    c1, c2, c3, c4 = x_1, fdot1, x_2, fdot2

    def f_one(h, c1, c2, c3, c4):
        return f(c1 - h * c2, c3 - h * c4)

    if case == 1:
        b, a = 100000, 0
        h = golden(f_one, b, a, c1, c2, c3, c4)

    elif case == 2:
        h = 0.001

    elif case == 3:
        alpha = 0.01
        h = 1
        q = 0.8
        while f(x_1 - h * fdot1, x_2 - h * fdot2) > f(x_1, x_2) - alpha * h * (fdot1 ** 2 + fdot2 ** 2) ** (1/2):
            h = q * h


    finish = f(x_1 - h * fdot1, x_2 - h * fdot2)
    x_s.append(x_1 - h * fdot1), y_s.append(x_2 - h * fdot2)
    if abs(start - finish) < 10 ** (-6):
        return x_1, x_2, finish

    return grad_1(f, fgrad, x_1 - h * fdot1, x_2 - h * fdot2, x_s, y_s, case)

def main():
    start_1, start_2 = 0, 0
    for case in (1, 2, 3):
        x_s = [start_1]
        y_s = [start_2]
        result = grad_1(func_2, func_2_grad, start_1, start_2, x_s, y_s, case)
        print(result)
        plt.plot(x_s, y_s)
        plt.show()


main()