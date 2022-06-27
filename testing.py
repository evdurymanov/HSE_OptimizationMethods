import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
rec = 100000
sys.setrecursionlimit(rec)

def i_ng_1(func, dots, counter, H_max_index, tau = 1.5):
    max_value, max_index = max([(y, -i) for i, (x, y, h) in enumerate(dots)])
    if abs(dots[max_index][0] - dots[max_index + 1][0]) < epsilon:
        func_values = [func(x) for (x, y, h) in dots]
        return min(func_values), counter, dots[H_max_index][2]
    L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]
    counter += 1
    new_dot = (dots[max_index][0] + dots[max_index + 1][0]) / 2 - \
              (func(dots[max_index + 1][0]) - func(dots[max_index][0])) / (2 * L)
    new_index = bisect.bisect_left(dots, (new_dot, 0))
    bisect.insort_left(dots, (new_dot, 0))
    H_1 = abs(func(dots[new_index][0]) - func(dots[new_index - 1][0])) / (dots[new_index][0] - dots[new_index - 1][0])
    H_2 = abs(func(dots[new_index + 1][0]) - func(dots[new_index][0])) / (dots[new_index + 1][0] - dots[new_index][0])

    dots[new_index - 1] = (dots[new_index - 1][0], 0, H_1)
    dots[new_index] = (new_dot, 0, H_2)
    max_value, H_max_index = max([(h, -i) for i, (x, y, h) in enumerate(dots)])
    H_max_index = H_max_index * (-1)
    L = 1 if dots[H_max_index][2] == 0 else tau * dots[H_max_index][2]


    new_char_1 = L * (dots[max_index][0] - dots[max_index - 1][0]) + ((func(dots[max_index][0]) - func(dots[max_index - 1][0])) ** 2) / (L * (dots[max_index][0] - dots[max_index - 1][0])) - 2 * (func(dots[max_index][0]) - func(dots[max_index - 1][0]))


    new_char_2 = L * (dots[max_index + 1][0] - dots[max_index][0]) + ((func(dots[max_index + 1][0]) - func(dots[max_index][0])) ** 2) / (L * (dots[max_index + 1][0] - dots[max_index][0])) - 2 * (func(dots[max_index + 1][0]) - func(dots[max_index][0]))



    dots[new_index - 1] = (dots[new_index - 1][0], new_char_1, H_1)
    dots[new_index] = (new_dot, new_char_2, H_2)

    return i_ng_1(func, dots, counter, H_max_index, tau)


for index, func in enumerate(test_functions):
    tau = 1.5
    a, b, counter = func.interval[0], func.interval[1], 0
    H = abs(func.f(b) - func.f(a)) / (b - a)
    L = 1 if H == 0 else tau * H
    epsilon = (b - a) * (10 ** -4)
    char_a = L * (b - a) + ((func.f(b) - func.f(a)) ** 2) / (L * (b - a)) - 2 * (func.f(b) - func.f(a))
    dots = [(a, char_a, H), (b, -float("inf"), -float("inf"))]
    maxval, count, H = i_ng_1(func.f, dots, counter, 0, tau)
    print(index + 1, maxval, count)