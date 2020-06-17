"""
Модуль реалізує методи одновимірної мінімізації
"""
import numpy as np


# методи нульового порядку
# дихотомія
def dihotomy(f, a, b, eps, delta):
    """Метод Дихотомії

    Args:
        f (function): функція
        a (int, float): ліва границя
        b (int, float): права границя
        eps (float): точність
        delta (float): мале додатнє число (< eps/2)

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    y = (a + b - delta)/2
    z = (a + b + delta)/2
    k = 0
    while abs(b - a) >= eps:
        if f(y) < f(z):
            b = z
        else:
            a = y
        y = (a + b - delta)/2
        z = (a + b + delta)/2
        k += 1
    return (a+b)/2, f((a+b)/2), k

# метод золотого перерізу
def golden_ration(f, a, b, eps):
    """Метод Золотого Перерізу

    Args:
        f (function): функція
        a (int, float): ліва границя
        b (int, float): права границя
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    y = a + ((3-np.sqrt(5))/2) * (b - a)
    z = a + b - y
    k = 0
    while np.abs(b - a) >= eps:
        if f(y) <= f(z):
            b = z
            z = y
            y = a + b - z
        else:
            a = y
            y = z
            z = a + b - z
        k += 1
    return (a+b)/2, f((a+b)/2), k

# метод квадратичної апроксимації
def powell(f, x1, h, eps):
    """Метод Квадратичної Апроксимації

    Args:
        f (function): функція
        x1 (int, float): початкова точка
        h (float): пробний крок
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    k = 0
    x2 = x1 + h
    if f(x1) > f(x2):
        x3 = x1 + 2*h
    else:
        x3 = x1 - h
    L = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
    f_min = L[0][0]
    x_min = L[0][1]
    a0 = f(x1)
    a1 = (f(x2) - f(x1)) / (x2 - x1)
    a2 = (1 / (x3 - x2)) * ((f(x3) - f(x1))/(x3 - x1) - (f(x2) - f(x1))/(x2 - x1))
    xx = (x2 + x1)/2 - a1/(2*a2)
    while np.abs(xx - x_min) >= eps:
        if f(xx) < f_min:
            x1 = xx
        else:
            x1 = x_min
            
        x2 = x1 + h
        if f(x1) > f(x2):
            x3 = x1 + 2*h
        else:
            x3 = x1 - h
        L = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
        f_min = L[0][0]
        x_min = L[0][1]
        a0 = f(x1)
        a1 = (f(x2) - f(x1)) / (x2 - x1)
        a2 = (1 / (x3 - x2)) * ((f(x3) - f(x1))/(x3 - x1) - (f(x2) - f(x1))/(x2 - x1))
        xx = (x2 + x1)/2 - a1/(2*a2)
        k += 1
    return xx, f(xx), k


# методи першого порядку
# метод середньої точки
def middle_point(f, df, a, b, eps):
    """Метод Cередньої Точки

    Args:
        f (function): функція
        df (function): диференційована функція
        a (int, float): ліва границя
        b (int, float): права границя
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    k = 0
    x = (a+b)/2
    d_f = df(x)
    while np.abs(d_f) > eps:
        if d_f > 0:
            b = x
        else:
            a = x
        x = (a+b)/2
        d_f = df(x)
        k = k + 1
    return x, f(x), k

# метод хорд
def chord(f, df, a, b, eps):
    """Метод Хорд

    Args:
        f (function): функція
        df (function): диференційована функція
        a (int, float): ліва границя
        b (int, float): права границя
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    k = 0
    x = a - df(a)/(df(a) - df(b)) * (a-b)
    d_f = df(x)
    while np.abs(d_f) > eps:
        if d_f > 0:
            b = x
        else:
            a = x
        x = a - df(a)/(df(a) - df(b)) * (a-b)
        d_f = df(x)
        k = k + 1
    return x, f(x), k

# методи другого порядку
# метод Нютона
def newton(f, df, d2f, x0, eps):
    """Метод Н'ютона

    Args:
        f (function): функція
        df (function): диференційована функція
        d2f (function): двічі диференційована функція
        x0 (int, float): початкова точка
        eps (float): точність

    Returns:
        list: список значень [координати мінімуму, значення функції в цій точці, кількість ітерацій]
    """
    d_f = df(x0)
    if np.abs(d_f) < eps:
        return x0, f(x0), 0
    d2_f = d2f(x0)
    x1 = x0 - d_f/d2_f
    d_f = df(x1)
    k = 1
    while np.abs(d_f) > eps:
        x0 = x1
        d2_f = d2f(x0)
        x1 = x0 - d_f/d2_f
        d_f = df(x1)
        k += 1
    return x1, f(x1), k

