import bisect
import sys
from test_functions import test_functions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
rec = 100000
sys.setrecursionlimit(rec)


def F(func, dots, x):
    f_s = [func(i) - tau * L * np.abs(x - i) for (i, _, _, _, L) in dots]
    f_s[0] = f_s[1] = f_s[-2] = f_s[-1] = -float("inf")
    return max(f_s)


def i_nl(func, dots, lam_max_ind, x_max_ind, counter):
    min_value, min_index = max([(y, -i) for i, (_, y, _, _, _) in enumerate(dots)])
    min_index *= -1

    if abs(dots[min_index][0] - dots[min_index + 1][0]) < epsilon_stop:
        func_values = [func(x) for (x, _, _, _, _) in dots[2:-2]]
        return min(func_values), counter

    counter += 1
    x_max_start = dots[x_max_ind + 1][0] - dots[x_max_ind][0]
    lam_max_start = (func(dots[lam_max_ind + 1][0]) - func(dots[lam_max_ind][0])) / (dots[lam_max_ind + 1][0] - dots[lam_max_ind][0])

    new_dot = ((dots[min_index][0] + dots[min_index + 1][0]) / 2) - ((func(dots[min_index + 1][0]) -
                                                                   func(dots[min_index][0])) / (2 * tau * dots[min_index][4]))

    new_index = bisect.bisect_left(dots, [new_dot, -float("inf"), -float("inf"), 0 , 0])

    bisect.insort_left(dots, [new_dot, -float("inf"), -float("inf"), 0 , 0])

    xs = []
    lambdas = []
    xs.append(-float("inf")), xs.append(-float("inf")), lambdas.append(-float("inf")), lambdas.append(-float("inf"))
    for i in range(2, len(dots) - 3):
        xs.append(dots[i + 1][0] - dots[i][0])
        lambdas.append(abs(func(dots[i + 1][0]) - func(dots[i][0])) / xs[-1])
    xs.append(-float("inf")), lambdas.append(-float("inf")), xs.append(-float("inf")), lambdas.append(-float("inf"))
    xs.append(-float("inf")), lambdas.append(-float("inf"))

    lambdas_1 = max(lambdas[new_index - 3], lambdas[new_index - 2], lambdas[new_index - 1])
    lambdas_2 = max(lambdas[new_index - 2], lambdas[new_index - 1], lambdas[new_index])
    lambdas_3 = max(lambdas[new_index - 1], lambdas[new_index], lambdas[new_index + 1])
    lambdas_4 = max(lambdas[new_index], lambdas[new_index + 1], lambdas[new_index + 2])

    ll = [0] * 4
    ll[0], ll[1], ll[2], ll[3] = lambdas_1, lambdas_2, lambdas_3, lambdas_4

    for i in (0, 1, 2, 3):
        if 1 < new_index - 2 + i < len(dots) - 3:
            dots[new_index - 2 + i][3] = ll[i]

    x_max_finish, x_max_ind = max((x, i) for i, x in enumerate(xs))
    lam_max_finish, lam_max_ind = max((lam, i) for i, lam in enumerate(lambdas))

    if x_max_start == x_max_finish and lam_max_start == lam_max_finish:
        gamma_1 = lam_max_finish * (dots[new_index][0] - dots[new_index - 1][0]) / x_max_start
        gamma_2 = lam_max_finish * (dots[new_index + 1][0] - dots[new_index][0]) / x_max_start
        gg = []
        gg.append(gamma_1), gg.append(gamma_2)
        HH = [0] * 4
        cc = [0] * 4


        H_1 = max(lambdas_1, dots[new_index - 2][2], epsilon)
        H_2 = max(lambdas_2, gamma_1, epsilon)
        H_3 = max(lambdas_3, gamma_2, epsilon)
        H_4 = max(lambdas_4, dots[new_index + 1][2], epsilon)

        HH[0], HH[1], HH[2], HH[3] = H_1, H_2, H_3, H_4

        char_1 = H_1 * (dots[new_index - 1][0] - dots[new_index - 2][0]) + ((
                        (func(dots[new_index - 1][0]) - func(dots[new_index - 2][0])) ** 2) / (
                                     H_1 * (dots[new_index - 1][0] - dots[new_index - 2][0]))) - 2 * (
                                     func(dots[new_index - 1][0]) + func(dots[new_index - 2][0]))

        char_2 = H_2 * (dots[new_index][0] - dots[new_index - 1][0]) + ((
                        (func(dots[new_index][0]) - func(dots[new_index - 1][0])) ** 2) / (
                                     H_2 * (dots[new_index][0] - dots[new_index - 1][0]))) - 2 * (
                                     func(dots[new_index][0]) + func(dots[new_index - 1][0]))

        char_3 = H_3 * (dots[new_index + 1][0] - dots[new_index][0]) + ((
                        (func(dots[new_index + 1][0]) - func(dots[new_index][0])) ** 2) / (
                                     H_3 * (dots[new_index + 1][0] - dots[new_index][0]))) - 2 * (
                                     func(dots[new_index + 1][0]) + func(dots[new_index][0]))

        char_4 = H_4 * (dots[new_index + 2][0] - dots[new_index + 1][0]) + ((
                        (func(dots[new_index + 2][0]) - func(dots[new_index + 1][0])) ** 2) / (
                                     H_4 * (dots[new_index + 2][0] - dots[new_index + 1][0]))) - 2 * (
                                     func(dots[new_index + 2][0]) + func(dots[new_index + 1][0]))

        cc[0], cc[1], cc[2], cc[3] = char_1, char_2, char_3, char_4


        for i in (0, 1 ,2 ,3):
            if 1 < new_index - 2 + i < len(dots) - 3:
                if i == 1 or i == 2:
                    dots[new_index - 2 + i] = [dots[new_index - 2 + i][0], cc[i], gg[i - 1], ll[i], HH[i]]
                else:
                    dots[new_index - 2 + i] = [dots[new_index - 2 + i][0], cc[i], dots[new_index - 2 + i][2], ll[i], HH[i]]


    else:
        gamma_s = []
        H_s = []
        char_s = []
        lambdaz = [lam for (_, _, _, lam, _) in dots]
        for i in range(2, len(dots) - 3):
            gamma_s.append((lam_max_finish / x_max_finish) * (dots[i + 1][0] - dots[i][0]))
            H_s.append(max(lambdaz[i], gamma_s[-1], epsilon))
            char_s.append(H_s[-1] * (dots[i + 1][0] - dots[i][0]) + ((
                        (func(dots[i + 1][0]) - func(dots[i][0])) ** 2) / (
                                     H_s[-1] * (dots[i + 1][0] - dots[i][0]))) - 2 * (
                                     func(dots[i + 1][0]) + func(dots[i][0])))
            dots[i] = [dots[i][0], char_s[-1], gamma_s[-1], lambdaz[i], H_s[-1]]

    return i_nl(func, dots, lam_max_ind, x_max_ind, counter)

epsilon = 10 ** (-6)
tau = 2




for index, func in enumerate(test_functions):
    a, b, L, counter = func.interval[0], func.interval[1], func.L, 0
    epsilon_stop = (b - a) * (10 ** -4)
    lam_max, x_max = abs(func.f(b) - func.f(a)) / (b - a), b - a
    H = max(lam_max * (b - a) / x_max, lam_max, epsilon)
    char = (H * (b - a)) + ((func.f(b) - func.f(a)) ** 2) / (H * (b - a)) - 2 * (func.f(b) + func.f(a))
    dots = [[-float("inf"), -float("inf"), -float("inf"), -float("inf"), float("inf")], [-float("inf"), -float("inf"), -float("inf"), -float("inf"), float("inf")], [a, char, lam_max * (b - a) / x_max, lam_max, H], [b, -float("inf"), -float("inf"), -float("inf"), -float("inf")], [float("inf"), -float("inf"), -float("inf"), -float("inf"), float("inf")], [float("inf"), -float("inf"), -float("inf"), -float("inf"), float("inf")]]
    minval, count = i_nl(func.f, dots, 2, 2, counter)
    print(index + 1, minval, count)





