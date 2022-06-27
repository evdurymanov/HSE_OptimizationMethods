import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
rec = 100000
sys.setrecursionlimit(rec)


def i_ng_2(func, dots, counter, H_max_index, tau):
    max_value, max_index = max([(y, -i) for i, (x, y, h) in enumerate(dots)])
    max_index *= -1
    H_start = dots[H_max_index][2]
    if abs(dots[max_index][0] - dots[max_index + 1][0]) < epsilon:
        func_values = [func(x) for (x, _, _) in dots]
        return min(func_values), counter, dots[H_max_index][2]
    L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]
    counter += 1
    new_dot = (dots[max_index][0] + dots[max_index + 1][0]) / 2 - \
              (func(dots[max_index + 1][0]) - func(dots[max_index][0])) / (2 * L)
    new_index = bisect.bisect_left(dots, (new_dot, 0, 0))
    bisect.insort_left(dots, (new_dot, 0, 0))
    H_1 = abs(func(dots[new_index][0]) - func(dots[new_index - 1][0])) / (dots[new_index][0] - dots[new_index - 1][0])
    H_2 = abs(func(dots[new_index + 1][0]) - func(dots[new_index][0])) / (dots[new_index + 1][0] - dots[new_index][0])
    dots[new_index - 1] = (dots[new_index - 1][0], 0, H_1)
    dots[new_index] = (new_dot, 0, H_2)
    max_value, H_max_index = max([(h, -i) for i, (x, y, h) in enumerate(dots)])
    H_max_index = H_max_index * (-1)
    L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]
    H_finish = dots[H_max_index][2]
    if H_start == H_finish:
        L_tild_1 = (L + (1 / L) * ((func(dots[new_index][0]) -
                                       func(dots[new_index - 1][0])) / (dots[new_index][0] - dots[new_index - 1][0])) ** 2) / 2
        R_1 = -4 * ((func(dots[new_index][0]) + func(dots[new_index - 1][0]) / 2) - L_tild_1 * ((dots[new_index][0] - dots[new_index - 1][0]) / 2))
        L_tild_2 = (L + (1 / L) * ((func(dots[new_index + 1][0]) -
                                    func(dots[new_index][0])) / (
                                               dots[new_index + 1][0] - dots[new_index][0])) ** 2) / 2
        R_2 = -4 * ((func(dots[new_index + 1][0]) + func(dots[new_index][0]) / 2) - L_tild_2 * (
                    (dots[new_index + 1][0] - dots[new_index][0]) / 2))
        dots[new_index - 1] = (dots[new_index - 1][0], R_1, H_1)
        dots[new_index] = (new_dot, R_2, H_2)
    else:
        for i in range(len(dots) - 1):
            L_tild = (L + (1 / L) * ((func(dots[i + 1][0]) -
                                        func(dots[i][0])) / (
                                                   dots[i + 1][0] - dots[i][0])) ** 2) / 2
            R = -4 * ((func(dots[i + 1][0]) + func(dots[i][0]) / 2) - L_tild * (
                    (dots[i + 1][0] - dots[i][0]) / 2))
            dots[i] = (dots[i][0], R, dots[i][2])

    return i_ng_2(func, dots, counter, H_max_index, tau)


for index, func in enumerate(test_functions):
    tau = 2
    a, b, counter = func.interval[0], func.interval[1], 0
    H = abs(func.f(b) - func.f(a)) / (b - a)
    L = 1 if H == 0 else tau * H
    epsilon = (b - a) * (10 ** -4)
    char_a = L * (b - a) + ((func.f(b) - func.f(a)) ** 2) / (L * (b - a)) - 2 * (func.f(b) + func.f(a))
    dots = [(a, char_a, H), (b, -float("inf"), -float("inf"))]
    maxval, count, H = i_ng_2(func.f, dots, counter, 0, tau)
    print(index + 1, maxval, count)

xarrays, yarrays, ogarrays = [], [], []
for index, func in enumerate(test_functions):
    x_arr = np.linspace(func.interval[0], func.interval[1], 1200)
    y, og = [], []
    tau = 1.5
    a, b, counter = func.interval[0], func.interval[1], 0
    H = abs(func.f(b) - func.f(a)) / (b - a)
    L = 1 if H == 0 else tau * H
    epsilon = (b - a) * (10 ** -4)
    char_a = L * (b - a) + ((func.f(b) - func.f(a)) ** 2) / (L * (b - a)) - 2 * (func.f(b) - func.f(a))
    dots = [(a, char_a, H), (b, -float("inf"), -float("inf"))]
    maxval, count, H = i_ng_2(func.f, dots, counter, H, tau)
    for i in x_arr:
        y.append(func.f(i))
        og.append(F(func.f, dots, H, i))
    xarrays.append(x_arr)
    yarrays.append(y)
    ogarrays.append(og)
