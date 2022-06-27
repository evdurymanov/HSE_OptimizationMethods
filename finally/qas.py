import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
rec = 100000
sys.setrecursionlimit(rec)



def func_1(x, n):
    return sum(100 * ((x[0] - x[1]) ** 2) + 5 * (1 - x[j]) ** 2 for j in range(1, n))

def func_1_grad(x, n):
    result = x * 10 - 10
    result[0], result[1] = 200 * x[0] - 200 * x[1], -200 * x[0] + 210 * x[1] - 10
    return np.array(result)

def grad(f, fgrad, d, gamma, H, it, case, n, counter):

    counter += 1

    if case == 1:
        temp_scalar = (d - H @ gamma) @ gamma
        if temp_scalar < 10 ** (-3) == 0:
            H_next = np.eye(n)
        else:
            H_next = H + (1 / temp_scalar) * np.outer((d - H @ gamma), np.transpose(d - H @ gamma))
        direction = H_next @ fgrad(it, n)
        if counter < 1:
            next_it = it - get_alpha(f, fgrad, it, direction) * direction
        else:
            next_it = it - H_next @ fgrad(it, n)
        d_next = next_it - it
        gamma_next = fgrad(next_it, n) - fgrad(it, n)

    elif case == 2:
        temp_1 = (1 / (gamma @ d)) * (np.outer(np.transpose(d), d))
        temp_2 = (1 / ((H @ gamma) @ gamma)) * (np.outer((H @ gamma), np.transpose(H @ gamma)))
        H_next = (H + temp_1 - temp_2)
        direction = H_next @ fgrad(it, n)
        if counter < 1:
            alpha = get_alpha(f, fgrad, it, direction)
            next_it = it - alpha * direction
        else:
            next_it = it - direction
        d_next = next_it - it
        gamma_next = fgrad(next_it, n) - fgrad(it, n)

    else:
        temp_1 = 1 / ((H @ gamma) @ gamma)
        temp_2 = np.outer(H @ gamma, np.transpose(d))
        temp_3 = np.outer(d, np.transpose(H @ gamma))
        temp_4 = 1 +((gamma @ d) / ((H @ gamma) @ gamma))
        temp_5 = np.outer((H @ gamma), np.transpose(H @ gamma))
        H_next = (H + temp_1 * (temp_2 + temp_3) - temp_1 * temp_4 * temp_5)
        direction = H_next @ fgrad(it, n)
        if counter < 1:
            next_it = it - get_alpha(f, fgrad, it, direction) * direction
        else:
             next_it = it - direction
        d_next = next_it - it
        gamma_next = fgrad(next_it, n) - fgrad(it, n)

    #xs.append(next_it[0]), ys.append(next_it[1])
    if abs(f(next_it, n) - f(it, n)) < 10 ** (5):
        return next_it, f(next_it, n), counter

    return grad(f, fgrad, d_next, gamma_next, H_next, next_it, case, n, counter)



n = 2
start = np.zeros(n)
for case in [1, 2, 3]:
    counter = 0
    H_0 = np.eye(n)
    direction = H_0 @ func_1_grad(start, n)
    it_1 = start - direction
    d = it_1 - start
    grad_it_1 = func_1_grad(it_1, n)
    gamma = grad_it_1 - func_1_grad(start, n)
    result = grad(func_1, func_1_grad, d, gamma, H_0, it_1, case, n, counter)
    print(result)