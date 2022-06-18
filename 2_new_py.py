import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
rec = 100000
sys.setrecursionlimit(rec)



def F(func, dots, L, x):
    f_s = [func(i) - L * np.abs(x - i) for (i, j) in dots]
    return max(f_s)


def ng(func, dots, counter, H_max_index, tau = 2):
    min_value, min_index = min([(y, i) for i, (x, y, h) in enumerate(dots)])
    if abs(dots[min_index][0] - dots[min_index + 1][0]) < epsilon:
        func_values = [func(x) for (x, y, h) in dots]
        return min(func_values), counter, dots[H_max_index][2]
    L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]
    counter += 1
    new_dot = (dots[min_index][0] + dots[min_index + 1][0]) / 2 - (func(dots[min_index][0]) -
                                                                   func(dots[min_index + 1][0])) / (2 * L)
    new_index = bisect.bisect_left(dots, (new_dot, 0, 0))
    bisect.insort_left(dots, (new_dot, 0, 0))
    H_1 = abs(func(dots[new_index][0]) - func(dots[new_index - 1][0])) / (dots[new_index][0] - dots[new_index - 1][0])
    H_2 = abs(func(dots[new_index + 1][0]) - func(dots[new_index][0])) / (dots[new_index + 1][0] - dots[new_index][0])
    dots[new_index - 1] = (dots[new_index - 1][0], 0, H_1)
    dots[new_index] = (new_dot, 0, H_2)
    if new_index - H_max_index == 1:
        max_value, H_max_index = max([(h, -i) for i, (x, y, h) in enumerate(dots)])
        H_max_index = H_max_index * (-1)
        L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]
    new_char_1 = (func(dots[new_index][0]) + func(dots[new_index - 1][0])) / 2 - \
                 L * (dots[new_index][0] - dots[new_index - 1][0])
    new_char_2 = (func(dots[new_index][0]) + func(dots[new_index + 1][0])) / 2 - \
                 L * (dots[new_index + 1][0] - dots[new_index][0])
    dots[new_index - 1] = (dots[new_index - 1][0], new_char_1, H_1)
    dots[new_index] = (new_dot, new_char_2, H_2)
    return ng(func, dots, counter, H_max_index, tau)


for index, func in enumerate(test_functions):
    tau = 5
    a, b, counter = func.interval[0], func.interval[1], 0
    H = abs(func.f(b) - func.f(a)) / (b - a)
    L = 1 if H == 0 else tau * H
    epsilon = (b - a) * (10 ** -4)
    char_a = (func.f(a) + func.f(b)) / 2 - L * (b - a) / 2
    dots = [(a, char_a, H), (b, float("inf"), -float("inf"))]
    minval, count, H = ng(func.f, dots, counter, 0, tau)
    print(index + 1, minval, count)