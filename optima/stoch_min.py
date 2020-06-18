"""
Модуль реалізує стохастичні методи мінімізації.
"""
import numpy as np
import math
import time
from random import random, uniform


# стохастичні методи
# метод імітації відпалу
def sim_anneal(f, x, a=-100, b=100):
    """Метод імітації відпалу

    Args:
        f (function): функція
        x (float): початкова точка
        a (int, optional): ліва границя
        b (int, optional): права границя

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """

    k = 1
    Tstart = 100
    Tend = 10**(-6)
    r = 0.98

    E = f(x)
    T = Tstart

    while T > Tend and k < 2000:
        # генерація нового випадкового стану
        x_new = x + 0.5*np.array([uniform(a/k, b/k), uniform(a/k, b/k)])
        E_new = f(x_new)
        deltaE = E_new - E
        if deltaE < 0:
            x = x_new
            E = E_new
            T = T * r
            k = k + 1
        else:
            # обчислення ймовірності прийняття нового стану
            p = np.exp(-deltaE / T)
            # генерація випадкового числа alpha з [0,1]
            alpha = random()
            if alpha < p:
                x = x_new
                E = E_new
                T = T * r
                k = k + 1
    return [x_new, f(x_new), k]

# метод рою частинок
def swarm_opt(f, s, d, xmin=-10, xmax=10, tmax=50):
    """Метод рою частинок

    Args:
        f (function): функція
        s (int): кількість частинок у рої
        d (int): кількість координат (розмірність)
        xmin (int, optional): нижня межа пошуку
        xmax (int, optional): верхня межа пошуку
        tmax (int, optional): час пошуку

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    x = np.zeros((s, d))
    v = np.zeros((s, d))
    p = np.zeros((s, d))
    fitness = np.zeros(s)
    fp = np.zeros(s)

    w = 1/(2*np.log(2))
    c1 = 0.5+np.log(2)
    c2 = 0.5+np.log(2)

    # ініціалізація рою
    for i in range(0, s):
        for j in range(0, d):
            x[i][j] = uniform(xmin, xmax)
            v[i][j] = (uniform(xmin, xmax) - x[i][j]) / 2
            p[i][j] = x[i][j]
        fitness[i] = f(x[i])
        fp[i] = fitness[i]

    # пошук кращої частинки у рої
    gbest = 0
    for i in range(1, s):
        if fp[i] < fp[gbest]:
            gbest = i
    k = 0
    # оновлення стану рою
    for _ in range(tmax):
        k += 1
        for i in range(0, s):
            for j in range(0, d):
                r1 = random()
                r2 = random()
                # визначення швидкості кожної частинки
                v[i][j] = w*v[i][j] + c1*r1*(p[i][j]-x[i][j])
                if (i != gbest):
                    v[i][j] = v[i][j] + c2*r2*(p[gbest][j] - x[i][j])
                # визначення нового положення кожної частинки
                x[i][j] = x[i][j] + v[i][j]
                # перевірка виходу за межі простору пошуку
                if x[i][j] < xmin:
                    x[i][j] = xmin
                    v[i][j] = 0
                if x[i][j] > xmax:
                    x[i][j] = xmax
                    v[i][j] = 0
        # оновлення поточного кращого положення частинки
        for i in range(0, s):
            fitness[i] = f(x[i])
            if fitness[i] < fp[i]:
                fp[i] = fitness[i]
                for j in range(0, d):
                    p[i][j] = x[i][j]
        # оновлення номеру кращої частинки
        for i in range(0, s):
            if fp[i] < fp[gbest]:
                gbest = i

#     plt.show()
    f_min = fp[gbest]
    x_min = p[gbest]
    return (x_min, f_min, k)


def wolf_opt(f, d, t_max=100, lb=-5, ub=5, s=50):
    """Метод оптимізації стаєю вовків

    Args:
        f (function): функція
        d (int): кількість координат (розмірність)
        t_max (int, optional): максимальна кількість ітерацій
        lb (int, optional): нижня границя
        ub (int, optional): верхня границя
        s (int, optional): кількість вовків у зграї
    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(d)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(d)
    Beta_score = float("inf")

    Delta_pos = np.zeros(d)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * d
    if not isinstance(ub, list):
        ub = [ub] * d

    # Initialize the positions of search agents
    Positions = np.zeros((s, d))
    for i in range(d):
        Positions[:, i] = np.random.uniform(
            0, 1, s) * (ub[i] - lb[i]) + lb[i]

    # Convergence_curve = np.zeros(Max_iter)

    # Main loop
    k = 0
    for l in range(0, t_max):
        k += 1
        for i in range(0, s):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(d):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = f(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2-l*((2)/t_max)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, s):
            for j in range(0, d):

                r1 = random()  # r1 is a random number in [0,1]
                r2 = random()  # r2 is a random number in [0,1]

                A1 = 2*a*r1-a  # Equation (3.3)
                C1 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 1
                D_alpha = abs(C1*Alpha_pos[j]-Positions[i, j])
                X1 = Alpha_pos[j]-A1*D_alpha  # Equation (3.6)-part 1

                r1 = random()
                r2 = random()

                A2 = 2*a*r1-a  # Equation (3.3)
                C2 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 2
                D_beta = abs(C2*Beta_pos[j]-Positions[i, j])
                X2 = Beta_pos[j]-A2*D_beta  # Equation (3.6)-part 2

                r1 = random()
                r2 = random()

                A3 = 2*a*r1-a  # Equation (3.3)
                C3 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 3
                D_delta = abs(C3*Delta_pos[j]-Positions[i, j])
                X3 = Delta_pos[j]-A3*D_delta  # Equation (3.5)-part 3

                Positions[i, j] = (X1+X2+X3)/3  # Equation (3.7)

    return Alpha_pos, Alpha_score, k
